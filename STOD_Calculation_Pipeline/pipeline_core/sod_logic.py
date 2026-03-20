# -*- coding: utf-8 -*-
"""
STOD Logic Core
==============
Single source of truth for STOD (Stability-Based):
- Reference (pure Python) versions for correctness checks and logs
- Numba-accelerated versions used by workers
"""

import numpy as np
from numba import njit

# Type codes
TYPE_T = 0   # Terminated
TYPE_UC = 1  # Unterminated with Cancels
TYPE_UU = 2  # Unterminated Uncanceled

TYPE_NAME = {
    TYPE_T:  "T",
    TYPE_UC: "UC",
    TYPE_UU: "UU",
}

# ------------------------------------------------------------------------------
# Reference (pure Python) - canonical semantics for validation & logs
# ------------------------------------------------------------------------------

def stod_pair_python(path_a, path_b):
    """
    Pure Python STOD core for one pair of paths.
    Returns:
        type_code       : int (0=T, 1=UC, 2=UU)
        total_score_unc : float
        terminated      : bool
        levels_data     : list of dicts
    """
    if len(path_a) == 0 or len(path_b) == 0:
        return 0, 0.0, False, []

    # Early same-cell termination semantics
    if (path_a[0][0] == path_b[0][0]) and (path_a[0][1] == path_b[0][1]):
        return 0, 0.0, True, []

    n = min(len(path_a), len(path_b))
    a_rows, a_cols = set(), set()
    b_rows, b_cols = set(), set()
    levels_data = []
    terminated = False

    # Track half-covered status: 0=none, 1=row covered (wait col), 2=col covered (wait row)
    a_status = []
    b_status = []

    # 1. Simulation Phase - O(n) optimized
    for level in range(n):
        rA, cA = int(path_a[level][0]), int(path_a[level][1])
        rB, cB = int(path_b[level][0]), int(path_b[level][1])

        # Check if new before adding
        b_row_is_new = rB not in b_rows
        b_col_is_new = cB not in b_cols
        a_row_is_new = rA not in a_rows
        a_col_is_new = cA not in a_cols

        a_rows.add(rA); a_cols.add(cA)
        b_rows.add(rB); b_cols.add(cB)

        levels_data.append({
            "level": level, "rA": rA, "cA": cA, "rB": rB, "cB": cB,
        })

        # Check if new B row/col completes any waiting A point
        if b_row_is_new:
            for i in range(level):
                if a_status[i] == 2 and levels_data[i]["rA"] == rB:
                    terminated = True
                    break
        if not terminated and b_col_is_new:
            for i in range(level):
                if a_status[i] == 1 and levels_data[i]["cA"] == cB:
                    terminated = True
                    break
        # Check if new A row/col completes any waiting B point
        if not terminated and a_row_is_new:
            for i in range(level):
                if b_status[i] == 2 and levels_data[i]["rB"] == rA:
                    terminated = True
                    break
        if not terminated and a_col_is_new:
            for i in range(level):
                if b_status[i] == 1 and levels_data[i]["cB"] == cA:
                    terminated = True
                    break

        if terminated:
            break

        # Register current point's half-covered status
        a_row_covered = rA in b_rows
        a_col_covered = cA in b_cols
        if a_row_covered and a_col_covered:
            terminated = True
            break
        elif a_row_covered:
            a_status.append(1)
        elif a_col_covered:
            a_status.append(2)
        else:
            a_status.append(0)

        b_row_covered = rB in a_rows
        b_col_covered = cB in a_cols
        if b_row_covered and b_col_covered:
            terminated = True
            break
        elif b_row_covered:
            b_status.append(1)
        elif b_col_covered:
            b_status.append(2)
        else:
            b_status.append(0)

    # 2. Scoring Phase
    total_score_unc = 0.0
    has_any_cancellation = False

    for d in levels_data:
        lvl = d["level"]
        rA, cA = d["rA"], d["cA"]
        rB, cB = d["rB"], d["cB"]
        
        canc_Ar = (rA in b_rows)
        canc_Ac = (cA in b_cols)
        canc_Br = (rB in a_rows)
        canc_Bc = (cB in a_cols)

        canc_count = int(canc_Ar) + int(canc_Ac) + int(canc_Br) + int(canc_Bc)
        unc_count  = 4 - canc_count
        contrib    = unc_count * lvl

        if canc_count > 0:
            has_any_cancellation = True

        d.update({
            "canc_Ar": canc_Ar, "canc_Ac": canc_Ac, 
            "canc_Br": canc_Br, "canc_Bc": canc_Bc,
            "canc_count": canc_count, 
            "uncanc_count": unc_count, 
            "unc_contrib": contrib
        })
        total_score_unc += contrib

    # 3. Determine Type
    if terminated:
        tcode = TYPE_T
    elif has_any_cancellation:
        tcode = TYPE_UC
    else:
        tcode = TYPE_UU

    return tcode, total_score_unc, terminated, levels_data


# ------------------------------------------------------------------------------
# Numba-accelerated (used by workers/canary)
# ------------------------------------------------------------------------------

@njit
def stod_pair_numba(path_a, path_b):
    """
    O(n) optimized STOD calculation.
    Returns: (type_code, total_score_unc)
    0 = T, 1 = UC, 2 = UU
    
    Optimization: Track points that are "half-covered" (row OR col matched).
    When a new row/col is added, only check if it completes any half-covered point.
    """
    len_a = path_a.shape[0]
    len_b = path_b.shape[0]

    if len_a == 0 or len_b == 0:
        return 0, 0.0

    # Early same-cell check
    if (path_a[0, 0] == path_b[0, 0]) and (path_a[0, 1] == path_b[0, 1]):
        return 0, 0.0

    n = len_a if len_a < len_b else len_b

    a_rows = set()
    a_cols = set()
    b_rows = set()
    b_cols = set()

    # Track "half-covered" status for each point:
    # 0 = not covered, 1 = row covered (waiting for col), 2 = col covered (waiting for row)
    a_status = np.zeros(n, dtype=np.int32)
    b_status = np.zeros(n, dtype=np.int32)

    terminated = False
    has_any_cancellation = False
    final_level = 0

    # 1. Simulation Loop - O(n) optimized
    for level in range(n):
        final_level = level

        rA = path_a[level, 0]
        cA = path_a[level, 1]
        rB = path_b[level, 0]
        cB = path_b[level, 1]

        # Check if this row/col is new before adding
        b_row_is_new = rB not in b_rows
        b_col_is_new = cB not in b_cols
        a_row_is_new = rA not in a_rows
        a_col_is_new = cA not in a_cols

        # Add to sets
        a_rows.add(rA)
        a_cols.add(cA)
        b_rows.add(rB)
        b_cols.add(cB)

        # If B added a new row, check all A points waiting for that row
        if b_row_is_new:
            for i in range(level):
                if a_status[i] == 2 and path_a[i, 0] == rB:  # waiting for row, and this is the row
                    terminated = True
                    break
        
        # If B added a new col, check all A points waiting for that col
        if not terminated and b_col_is_new:
            for i in range(level):
                if a_status[i] == 1 and path_a[i, 1] == cB:  # waiting for col, and this is the col
                    terminated = True
                    break

        # If A added a new row, check all B points waiting for that row
        if not terminated and a_row_is_new:
            for i in range(level):
                if b_status[i] == 2 and path_b[i, 0] == rA:  # waiting for row, and this is the row
                    terminated = True
                    break

        # If A added a new col, check all B points waiting for that col
        if not terminated and a_col_is_new:
            for i in range(level):
                if b_status[i] == 1 and path_b[i, 1] == cA:  # waiting for col, and this is the col
                    terminated = True
                    break

        if terminated:
            break

        # Register current A point's status
        a_row_covered = rA in b_rows
        a_col_covered = cA in b_cols
        if a_row_covered and a_col_covered:
            terminated = True
            break
        elif a_row_covered:
            a_status[level] = 1  # row covered, waiting for col
        elif a_col_covered:
            a_status[level] = 2  # col covered, waiting for row
        # else: neither covered, status stays 0

        # Register current B point's status
        b_row_covered = rB in a_rows
        b_col_covered = cB in a_cols
        if b_row_covered and b_col_covered:
            terminated = True
            break
        elif b_row_covered:
            b_status[level] = 1  # row covered, waiting for col
        elif b_col_covered:
            b_status[level] = 2  # col covered, waiting for row
            
    # 2. Scoring Loop
    total_score = 0.0
    for level in range(final_level + 1):
        rA = path_a[level, 0]
        cA = path_a[level, 1]
        rB = path_b[level, 0]
        cB = path_b[level, 1]

        canc_Ar = rA in b_rows
        canc_Ac = cA in b_cols
        canc_Br = rB in a_rows
        canc_Bc = cB in a_cols
        
        canc = 0
        if canc_Ar: canc += 1
        if canc_Ac: canc += 1
        if canc_Br: canc += 1
        if canc_Bc: canc += 1
        
        if canc > 0:
            has_any_cancellation = True

        unc = 4 - canc
        total_score += unc * level

    # 3. Determine Type
    if terminated:
        tcode = 0 # TYPE_T
    elif has_any_cancellation:
        tcode = 1 # TYPE_UC
    else:
        tcode = 2 # TYPE_UU

    return tcode, total_score
