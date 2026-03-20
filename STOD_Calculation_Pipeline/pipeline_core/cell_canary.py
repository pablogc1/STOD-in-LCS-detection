# -*- coding: utf-8 -*-
"""
Cell Canary Test
================
Tests specific cells (matching Python code) with detailed logging
to compare pipeline results with Python results.
"""
import os
import sys
import yaml
import numpy as np
import importlib
from scipy.integrate import odeint

# Import STOD Logic
from pipeline_core.sod_logic import stod_pair_python, TYPE_NAME

# ----------------------------------------------------------------------
# Configuration & Setup
# ----------------------------------------------------------------------

def load_config():
    path = os.environ.get('SYSTEM_CONFIG_FILE')
    if not path or not os.path.exists(path):
        print("ERROR: SYSTEM_CONFIG_FILE not set or invalid.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_system_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"ERROR: Could not import system module '{module_name}': {e}")
        sys.exit(1)

def format_stod_log(levels_data, total_score_unc, label="STOD pair"):
    """
    Format matching Python code format exactly.
    """
    lines = [f"=== {label} ===", "--- STOD Detailed Uncanceled Contribution Log ---\n"]
    header = f"{'Level':^5} | {'Side A (r,c)':^20} | {'Side B (r,c)':^20} | {'Canc':^4} | {'Unc':^4} | {'Unc * level':^18}"
    lines.append(header)
    lines.append("-" * len(header))

    for d in levels_data:
        lvl = d["level"]
        # Wrap in parentheses if that specific coordinate was found in the other path's history
        sA_r = f"({d['rA']:>2})" if d["canc_Ar"] else f" {d['rA']:>2} "
        sA_c = f"({d['cA']:>2})" if d["canc_Ac"] else f" {d['cA']:>2} "
        sB_r = f"({d['rB']:>2})" if d["canc_Br"] else f" {d['rB']:>2} "
        sB_c = f"({d['cB']:>2})" if d["canc_Bc"] else f" {d['cB']:>2} "
        
        lines.append(
            f"{lvl:^5} | {f'({sA_r}, {sA_c})':^20} | {f'({sB_r}, {sB_c})':^20} | "
            f"{d['canc_count']:^4d} | {d['uncanc_count']:^4d} | {d['uncanc_count']} * {lvl} = {d['unc_contrib']:.1f}"
        )

    lines.append(f"\nTotal (Σ Unc * level) = {total_score_unc:.1f}")
    return "\n".join(lines)

def build_time_array(t_start, t_end, dt):
    """Build time array matching Python code."""
    if abs(t_end - t_start) < 1e-12:
        return np.array([t_start], dtype=float)
    step = dt if (t_end > t_start) else -dt
    return np.arange(t_start, t_end + step/100.0, step, dtype=float)

def get_path_restart(sys_mod, r, c, t_snap, t_end, dt, grid_params, phys_params, direction):
    """
    Get path starting from t_snap, matching Python code exactly.
    direction: "fwd" or "bwd"
    """
    # Get initial state at grid cell
    s0 = sys_mod.get_initial_state_from_grid(r, c, grid_params)
    if s0 is None:
        return None
    
    # Build time array
    if direction == "fwd":
        t_eval = build_time_array(t_snap, t_end, dt)
    else:  # bwd
        t_eval = build_time_array(t_snap, 0.0, dt)
    
    # Integrate
    def func(y, t):
        return sys_mod.get_velocity_field(y, t, phys_params)
    traj = odeint(func, s0, t_eval)
    
    # Discretize
    grid_path_tuples = sys_mod.discretize_trajectory_to_grid(traj, grid_params)
    path = np.array(grid_path_tuples, dtype=np.int64)
    
    # Ensure first point is the starting cell (matching Python: path[0] = [r, c])
    if len(path) > 0:
        path[0] = np.array([r, c], dtype=np.int64)
    
    return path

def run_cell_canary():
    """
    Run canary test on specific cells matching Python code.
    Tests cell (62, 125) at various snapshot times.
    """
    print("=" * 80)
    print("CELL CANARY TEST (Matching Python Code)")
    print("=" * 80)
    
    cfg = load_config()
    
    # Setup System
    sys_mod_name = cfg['system_module']
    sys_mod = load_system_module(sys_mod_name)
    phys_params = cfg['physics_params']
    grid_params = cfg['grid_params']
    time_params = cfg['time_params']
    
    # Parameters matching Python code
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    t_end = float(time_params['total_integration_time_end'])
    dt = float(time_params['time_step_dt'])
    
    # Snapshot times (matching Python code)
    snaps = time_params.get('snapshot_times', [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    directions = ["fwd", "bwd"]
    
    # Cell to test (matching Python: CELLS_TO_INSPECT)
    cells_to_inspect = [(62, 125)]  # Matching the Python canary
    
    # Output folder
    output_base = cfg['output_base_folder']
    output_folder = os.path.join(output_base, "Pipeline Code Canary Cell Results")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"System: {sys_mod_name}")
    print(f"Grid: {nx}x{ny}")
    print(f"Output folder: {output_folder}")
    print(f"Cells to inspect: {cells_to_inspect}")
    print(f"Snapshot times: {snaps}")
    print()
    
    for r, c in cells_to_inspect:
        for t_snap in snaps:
            for direction in directions:
                print(f"Processing: Cell ({r},{c}) | Dir: {direction} | t: {t_snap}")
                
                paths_i, paths_s = {}, {}
                targets = [
                    (r, c, 'CENTER'),
                    (r+1, c, 'UP'),
                    (r-1, c, 'DOWN'),
                    (r, c-1, 'LEFT'),
                    (r, c+1, 'RIGHT')
                ]
                
                # Generate paths for center and neighbors
                for nr, nc, key in targets:
                    if not (0 <= nr < ny and 0 <= nc < nx):
                        continue
                    p = get_path_restart(sys_mod, nr, nc, t_snap, t_end, dt, grid_params, phys_params, direction)
                    if p is not None:
                        paths_i[key] = p
                        paths_s[key] = p[::-1]  # STOD is just the reversed array
                
                # Reversibility check
                is_reversed = False
                if 'CENTER' in paths_i and 'CENTER' in paths_s:
                    is_reversed = np.array_equal(paths_i['CENTER'], paths_s['CENTER'][::-1])
                
                # Build log
                log_header = [
                    f"CELL NEIGHBORHOOD LOG: Center ({r},{c}) | Dir: {direction} | t: {t_snap}",
                    f"REVERSIBILITY CHECK: Center path STOD == Center path ISTOD[::-1]? {is_reversed}",
                    "="*75,
                    ""
                ]
                
                neighbor_logs = []
                for n_key in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    if n_key in paths_i and 'CENTER' in paths_i:
                        ti, si, term_i, li = stod_pair_python(paths_i['CENTER'], paths_i[n_key])
                        ts, ss, term_s, ls = stod_pair_python(paths_s['CENTER'], paths_s[n_key])
                        
                        neighbor_logs.append(f">>> COMPARISON WITH NEIGHBOR: {n_key}")
                        neighbor_logs.append(format_stod_log(li, si, label=f"ISTOD ({n_key})"))
                        neighbor_logs.append(format_stod_log(ls, ss, label=f"STOD Rev ({n_key})"))
                        neighbor_logs.append("-" * 50)
                
                # Write log file
                log_filename = f"log_nb_r{r}_c{c}_{direction}_t{t_snap}.txt"
                log_path = os.path.join(output_folder, log_filename)
                with open(log_path, "w") as f:
                    f.write("\n".join(log_header + neighbor_logs))
                
                print(f"  -> Saved: {log_filename}")
    
    print()
    print("=" * 80)
    print(f"Canary test complete! Results saved in: {output_folder}")
    print("=" * 80)

if __name__ == "__main__":
    run_cell_canary()


