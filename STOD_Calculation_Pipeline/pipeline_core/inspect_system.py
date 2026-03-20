# -*- coding: utf-8 -*-
"""
System Inspection Canary
========================
Runs a full integration and STOD check on a single specific cell of the 
configured system (e.g., LiCN center). Generates a detailed text log.
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

def format_stod_log(levels_data, total_score_unc, label):
    lines = []
    lines.append(f"=== {label} ===")
    lines.append("--- STOD Uncanceled Contribution Log ---\n")
    header = (
        f"{'Level':^5} | {'Side A (r,c)':^20} | "
        f"{'Side B (r,c)':^20} | {'Canc':^4} | {'Unc':^4} | {'Unc * level':^18}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for d in levels_data:
        lvl = d["level"]
        # Determine cancellation status for display
        sA_r = f"({d['rA']})" if d["canc_Ar"] else f" {d['rA']} "
        sA_c = f"({d['cA']})" if d["canc_Ac"] else f" {d['cA']} "
        sB_r = f"({d['rB']})" if d["canc_Br"] else f" {d['rB']} "
        sB_c = f"({d['cB']})" if d["canc_Bc"] else f" {d['cB']} "

        lines.append(
            f"{lvl:^5} | {f'({sA_r},{sA_c})':^20} | {f'({sB_r},{sB_c})':^20} | "
            f"{d['canc_count']:^4d} | {d['uncanc_count']:^4d} | "
            f"{d['uncanc_count']} * {lvl} = {d['unc_contrib']:.1f}"
        )

    lines.append("")
    lines.append(f"Total Stability Score: {total_score_unc:.4f}")
    return "\n".join(lines)

# ----------------------------------------------------------------------
# Discretization
# ----------------------------------------------------------------------

def discretize(trajectory, grid_params):
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    y_min, y_max = float(grid_params['y_min']), float(grid_params['y_max'])
    
    # Simple Generic Projection (taking first 2 cols)
    # Note: For LiCN, we need the specific projection logic if the system returns 4D.
    # We rely on the python system module having 'discretize_trajectory_to_grid'
    # IF available, otherwise we do simple binning.
    
    # However, to be robust with the C++ pipeline logic, we should ideally use
    # the system module's discretize function if it exists.
    return trajectory # Placeholder, handled in main loop via system module

# ----------------------------------------------------------------------
# Main Inspection Logic
# ----------------------------------------------------------------------

def main():
    print("--- Starting System Inspection Canary ---")
    cfg = load_config()
    
    # 1. Setup System
    sys_mod_name = cfg['system_module']
    sys_mod = load_system_module(sys_mod_name)
    phys_params = cfg['physics_params']
    grid_params = cfg['grid_params']
    time_params = cfg['time_params']
    
    # 2. Select Cell (Center of grid)
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    r_center, c_center = ny // 2, nx // 2
    
    print(f"System: {sys_mod_name}")
    print(f"Inspecting Center Cell: Grid(row={r_center}, col={c_center})")
    
    # 3. Setup Output
    out_folder = cfg['output_base_folder']
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "system_inspection_log.txt")
    
    log_buffer = []
    log_buffer.append(f"System Inspection Log for {sys_mod_name}")
    log_buffer.append(f"Physics: {phys_params}")
    log_buffer.append(f"Target Cell: ({r_center}, {c_center})")
    log_buffer.append("="*60 + "\n")

    # 4. Integrate & Analyze (Forward Only for brevity, or both)
    # We will do Forward analysis for the first snapshot time (usually 0.0)
    t_start = 0.0
    t_end = float(time_params['total_integration_time_end'])
    dt = float(time_params['time_step_dt'])
    
    t_eval = np.arange(t_start, t_end + dt/100.0, dt)
    
    # Helper to get path
    def get_grid_path(r, c):
        # 1. Grid -> State
        s0 = sys_mod.get_initial_state_from_grid(r, c, grid_params)
        if s0 is None: return None
        
        # 2. Integrate
        # Wrapper for odeint
        def func(y, t): return sys_mod.get_velocity_field(y, t, phys_params)
        traj = odeint(func, s0, t_eval)
        
        # 3. Discretize
        # Use system module discretization
        grid_path_tuples = sys_mod.discretize_trajectory_to_grid(traj, grid_params)
        return np.array(grid_path_tuples, dtype=np.int32)

    # Get Center Path
    path_center = get_grid_path(r_center, c_center)
    
    if path_center is None:
        log_buffer.append("Center cell is INVALID (Forbidden region).")
    else:
        # Check Neighbors
        neighbors = [(-1,0, "DOWN"), (1,0, "UP"), (0,-1, "LEFT"), (0,1, "RIGHT")]
        
        for dr, dc, name in neighbors:
            nr, nc = r_center + dr, c_center + dc
            path_neigh = get_grid_path(nr, nc)
            
            label = f"FWD Center vs {name} ({nr},{nc})"
            log_buffer.append(f"\n--- {label} ---")
            
            if path_neigh is None:
                log_buffer.append("Neighbor is INVALID.")
                continue
                
            # Run STOD
            tcode, score, term, levels = stod_pair_python(path_center, path_neigh)
            
            log_buffer.append(f"Result Type: {TYPE_NAME[tcode]} ({tcode})")
            log_buffer.append(f"Score: {score:.4f}")
            log_buffer.append(f"Terminated: {term}")
            log_buffer.append(format_stod_log(levels, score, label))

    # 5. Write to file
    with open(out_file, 'w') as f:
        f.write("\n".join(log_buffer))
        
    print(f"Inspection log saved to: {out_file}")

if __name__ == "__main__":
    main()
