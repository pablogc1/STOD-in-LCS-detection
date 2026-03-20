import numpy as np
import sys
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# STOD Logic (embedded to avoid numba dependency)
# Type codes
TYPE_T = 0   # Terminated
TYPE_UC = 1  # Unterminated with Cancels
TYPE_UU = 2  # Unterminated Uncanceled

TYPE_NAME = {
    TYPE_T:  "T",
    TYPE_UC: "UC",
    TYPE_UU: "UU",
}

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

# --- REUSING LOGIC FROM CANARIES ---

def gen_serpentine(s=6):
    """Generates coordinates for a serpentine path (size s x s)."""
    path = []
    for r in range(s):
        cols = range(s) if r % 2 == 0 else range(s - 1, -1, -1)
        for c in cols:
            path.append((r, c))
    return np.array(path, dtype=np.int64)

def cyclic_slice(full_path, start_idx, length):
    """Take 'length' points along the serpentine cycle starting at start_idx."""
    N = full_path.shape[0]
    idxs = (start_idx + np.arange(length)) % N
    return full_path[idxs]

def format_stod_log(levels_data, total_score_unc, label="STOD pair"):
    """Format the detailed STOD calculation log as seen in the canary."""
    lines = []
    lines.append(f"=== {label} ===")
    lines.append("--- STOD Detailed Uncanceled Contribution Log ---\n")
    header = (
        f"{'Level':^5} | {'Side A (r,c)':^20} | "
        f"{'Side B (r,c)':^20} | {'Canc':^4} | {'Unc':^4} | {'Unc * level':^18}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for d in levels_data:
        lvl = d["level"]
        rA, cA = d["rA"], d["cA"]
        rB, cB = d["rB"], d["cB"]

        canc_count = d["canc_count"]
        unc_count  = d["uncanc_count"]
        contrib    = d["unc_contrib"]

        # Parenthesis around canceled row / col
        sA_r = f"({rA:>2})" if d["canc_Ar"] else f" {rA:>2} "
        sA_c = f"({cA:>2})" if d["canc_Ac"] else f" {cA:>2} "
        sB_r = f"({rB:>2})" if d["canc_Br"] else f" {rB:>2} "
        sB_c = f"({cB:>2})" if d["canc_Bc"] else f" {cB:>2} "

        sideA_str = f"({sA_r}, {sA_c})"
        sideB_str = f"({sB_r}, {sB_c})"

        lines.append(
            f"{lvl:^5} | {sideA_str:^20} | {sideB_str:^20} | "
            f"{canc_count:^4d} | {unc_count:^4d} | "
            f"{unc_count} * {lvl} = {contrib:.1f}"
        )

    lines.append("")
    lines.append(f"Total (Σ Unc * level) = {total_score_unc:.1f}")
    return "\n".join(lines)

# --- PLOTTING LOGIC (Matching 'Snake and SOD.py' aesthetics) ---

def plot_snake_with_points(full_path, grid_size, idx_A, idx_B, length, filename="serpentine_instod_finstod_plot.pdf"):
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Grid lines (Match older style)
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal', adjustable='box')

    # Axis labels bigger
    ax.set_xlabel('Column Index', fontsize=16)
    ax.set_ylabel('Row Index', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    # Serpentine path (red, background)
    rows = full_path[:, 0]
    cols = full_path[:, 1]
    ax.plot(cols, rows, color='red', linewidth=2, alpha=0.8, zorder=1, label='Full Path')

    # Trajectories
    traj_A = cyclic_slice(full_path, idx_A, length)
    traj_B = cyclic_slice(full_path, idx_B, length)

    # Colors
    color_A = 'blue'
    color_B = 'limegreen' # Lighter green for better overlap visibility

    # Plot Trajectory A (blue)
    ax.plot(traj_A[:, 1], traj_A[:, 0], color=color_A, linewidth=4, alpha=0.7, zorder=2, label='Trajectory A')
    # Plot Trajectory B (green)
    ax.plot(traj_B[:, 1], traj_B[:, 0], color=color_B, linewidth=4, alpha=0.7, zorder=2, label='Trajectory B')

    # Add arrows for direction (Everywhere on trajectories)
    def add_arrows(traj, color, lw=2, alpha=0.8):
        for i in range(len(traj) - 1):
            start = traj[i]
            end = traj[i+1]
            # Small arrow
            ax.annotate('', xy=(end[1], end[0]), xytext=(start[1], start[0]),
                        arrowprops=dict(arrowstyle='->', lw=lw, color=color, alpha=alpha),
                        zorder=3)

    # Arrows for full path (red)
    add_arrows(full_path, 'red', lw=1, alpha=0.6)
    # Arrows for A and B
    add_arrows(traj_A, color_A, lw=2, alpha=0.9)
    add_arrows(traj_B, color_B, lw=2, alpha=0.9)

    # Point A_0 (Initial Condition)
    pA0 = traj_A[0]
    ax.plot(pA0[1], pA0[0], 'o', markersize=35, color=color_A, markeredgecolor='black', markeredgewidth=3, zorder=4)
    txtA0 = ax.text(pA0[1], pA0[0], r'$A_0$', ha='center', va='center', color='white', fontweight='bold', fontsize=18, zorder=5)
    import matplotlib.patheffects as path_effects
    txtA0.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    # Point A_tau (Final State)
    pAf = traj_A[-1]
    ax.plot(pAf[1], pAf[0], 'o', markersize=35, color=color_A, markeredgecolor='black', markeredgewidth=3, zorder=4)
    txtAf = ax.text(pAf[1], pAf[0], r'$A_\tau$', ha='center', va='center', color='white', fontweight='bold', fontsize=18, zorder=5)
    txtAf.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    # Point B_0 (Initial Condition)
    pB0 = traj_B[0]
    ax.plot(pB0[1], pB0[0], 'o', markersize=35, color=color_B, markeredgecolor='black', markeredgewidth=3, zorder=4)
    txtB0 = ax.text(pB0[1], pB0[0], r'$B_0$', ha='center', va='center', color='white', fontweight='bold', fontsize=18, zorder=5)
    txtB0.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    # Point B_tau (Final State)
    pBf = traj_B[-1]
    ax.plot(pBf[1], pBf[0], 'o', markersize=35, color=color_B, markeredgecolor='black', markeredgewidth=3, zorder=4)
    txtBf = ax.text(pBf[1], pBf[0], r'$B_\tau$', ha='center', va='center', color='white', fontweight='bold', fontsize=18, zorder=5)
    txtBf.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    # Legend
    ax.legend(loc='upper right', fontsize=12)

    # Save PDF
    fig.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to: {filename}")

    # Legend
    ax.legend(loc='upper right', fontsize=12)

    # Save PDF
    fig.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to: {filename}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Standard Canary Parameters
    GRID_SIZE = 6
    TIME_UNITS = 10
    STEPS = TIME_UNITS
    LENGTH = STEPS + 1
    IDX_A, IDX_B = 5, 10

    print("=" * 80)
    print("      STOD VISUALIZATION: STOD (Forward) & FinSTOD (Reversed)")
    print("=" * 80)

    # 1. Setup paths
    full_snake = gen_serpentine(GRID_SIZE)
    traj_A = cyclic_slice(full_snake, IDX_A, LENGTH)
    traj_B = cyclic_slice(full_snake, IDX_B, LENGTH)
    
    # Reversed trajectories for FinSTOD
    traj_A_rev = traj_A[::-1].copy()
    traj_B_rev = traj_B[::-1].copy()

    # 2. Compute
    ti, si, term_i, li = stod_pair_python(traj_A, traj_B)
    ts, ss, term_s, ls = stod_pair_python(traj_A_rev, traj_B_rev)

    # 3. Print Results
    print(f"\nConfiguration: Grid {GRID_SIZE}x{GRID_SIZE} | Path Length {LENGTH}")
    print(f"Start A: Index {IDX_A} -> {full_snake[IDX_A]}")
    print(f"Start B: Index {IDX_B} -> {full_snake[IDX_B]}")
    
    print("\n" + "="*80)
    print("STOD LOG (Forward)")
    print("="*80)
    print(format_stod_log(li, si, label=f"STOD - {TYPE_NAME[ti]}"))
    
    print("\n" + "="*80)
    print("FinSTOD LOG (Reversed)")
    print("="*80)
    print(format_stod_log(ls, ss, label=f"FinSTOD - {TYPE_NAME[ts]}"))

    # 4. Plot
    plot_snake_with_points(full_snake, GRID_SIZE, IDX_A, IDX_B, LENGTH)
