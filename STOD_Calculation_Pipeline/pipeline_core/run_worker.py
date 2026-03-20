import os
import sys
import argparse
import yaml
import subprocess

def get_system_shortname(module_str):
    if "." in module_str:
        return module_str.split(".")[-1]
    return module_str

def collect_numeric_params(config):
    """
    Scans physics_params, grid_params, and time_params.
    Returns a list of 'key=value' strings for ALL numeric parameters found.
    """
    params = []
    sections = ['physics_params', 'grid_params', 'time_params']
    for sec in sections:
        if sec in config and isinstance(config[sec], dict):
            for key, val in config[sec].items():
                if isinstance(val, bool): continue
                if isinstance(val, (int, float)):
                    params.append(f"{key}={val}")
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True)
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = os.environ.get('SYSTEM_CONFIG_FILE')
    if not config_path:
        sys.exit("ERROR: SYSTEM_CONFIG_FILE not set")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Paths
    out_base = config['output_base_folder']
    # Note: In Pipeline 1, we use "partial" for input, and "partial_results" for output
    traj_folder = os.path.join(out_base, "trajectory_cache", "partial")
    res_dir = os.path.join(out_base, "partial_results")
    os.makedirs(res_dir, exist_ok=True)

    # Input file (Chunk from GenTraj)
    input_h5 = os.path.join(traj_folder, f"chunk_{args.worker_id:04d}.h5")
    
    if not os.path.exists(input_h5):
        # Graceful exit if job array is larger than rows
        print(f"Worker {args.worker_id}: Input {input_h5} not found (likely empty row range). Exiting.")
        sys.exit(0)

    # 3. Calculate Row Range (Critical for Fresh FTLE/LB absolute Y coords)
    gp = config['grid_params']
    ny = int(gp['grid_resolution_y'])
    
    # We rely on SLURM_ARRAY_TASK_COUNT passed by the master script
    total_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))
    
    rows_per_job = (ny + total_jobs - 1) // total_jobs
    start_row = (args.worker_id - 1) * rows_per_job
    end_row = min(ny, start_row + rows_per_job)

    if start_row >= end_row:
        print(f"Worker {args.worker_id} has no rows to process. Exiting.")
        sys.exit(0)

    # 4. Binary Path
    binary = "./cpp_backend/calc_metrics"
    if not os.path.exists(binary):
        print(f"ERROR: Binary not found at {binary}")
        sys.exit(1)

    # 5. Prepare Common Arguments
    sys_name = get_system_shortname(config['system_module'])
    tp = config['time_params']
    snaps = tp['snapshot_times']
    extra_params = collect_numeric_params(config)

    print(f"--- Worker {args.worker_id} (Rows {start_row}-{end_row}) ---")

    # Get analysis controls
    ac = config.get('analysis_controls', {})
    
    # Get field mode (orthogonal option)
    # Check for environment variable override (used by "both" mode in master_run.sh)
    field_mode_override = os.environ.get("FIELD_MODE_OVERRIDE")
    if field_mode_override:
        field_mode = field_mode_override
        print(f"[INFO] Using FIELD_MODE_OVERRIDE: {field_mode}")
    else:
        field_mode = config.get('field_mode', 'standard')
    
    use_orthogonal = (field_mode == 'orthogonal')
    
    # FTLE method selection: variational (accurate but slow) vs finite-diff (faster)
    use_variational_ftle = ac.get('use_variational_ftle', True)  # Default to variational for backward compat
    print(f"[INFO] FTLE method: {'variational' if use_variational_ftle else 'finite-difference'}")
    
    # =========================================================
    # PART A: Local Metrics (Using Pre-computed Trajectories)
    # =========================================================
    has_local = (ac.get('compute_ftle_forward', False) or ac.get('compute_ftle_backward', False) or
                 ac.get('compute_local_stod_forward', False) or ac.get('compute_local_stod_backward', False) or
                 ac.get('compute_local_finstod_forward', False) or ac.get('compute_local_finstod_backward', False) or
                 ac.get('compute_fli_forward', False) or ac.get('compute_fli_backward', False) or
                 ac.get('compute_ld_forward', False) or ac.get('compute_ld_backward', False))
    
    if has_local:
        for t_snap in snaps:
            t_val = float(t_snap)
            out_name = f"results_chunk_{args.worker_id:04d}_snap_{t_val:.2f}.h5"
            out_path = os.path.join(res_dir, out_name)
            
            # Args must match the new C++ main() signature:
            # 1:sys, 2:in, 3:out, 4:nx, 5:ny, 6:x_min, 7:x_max, 8:y_min, 9:y_max, 
            # 10:t_snap, 11:t_total, 12:start_row, 13:end_row
            cmd = [
                binary,
                sys_name,
                input_h5,
                out_path,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                str(t_val),
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row)
            ]
            
            # Add analysis control flags
            if ac.get('compute_ftle_forward', False):
                cmd.append('--compute-ftle-forward')
            if ac.get('compute_ftle_backward', False):
                cmd.append('--compute-ftle-backward')
            if ac.get('compute_local_stod_forward', False):
                cmd.append('--compute-stod-forward')
            if ac.get('compute_local_stod_backward', False):
                cmd.append('--compute-stod-backward')
            if ac.get('compute_local_finstod_forward', False):
                cmd.append('--compute-finstod-forward')
            if ac.get('compute_local_finstod_backward', False):
                cmd.append('--compute-finstod-backward')
            
            # FLI (Fast Lyapunov Indicator) flags
            if ac.get('compute_fli_forward', False):
                cmd.append('--compute-fli-forward')
            if ac.get('compute_fli_backward', False):
                cmd.append('--compute-fli-backward')
            fli_alpha = ac.get('fli_alpha_rescale', 0.0)
            if fli_alpha > 0:
                cmd.append(f'fli_alpha={fli_alpha}')
            
            # LD (Lagrangian Descriptor) flags
            if ac.get('compute_ld_forward', False):
                cmd.append('--compute-ld-forward')
            if ac.get('compute_ld_backward', False):
                cmd.append('--compute-ld-backward')
            if ac.get('compute_ld_forward', False) or ac.get('compute_ld_backward', False):
                ld_p = ac.get('ld_p_norm', 0.5)
                cmd.append(f'ld_p_norm={ld_p}')
            
            # Orthogonal field mode
            if use_orthogonal:
                cmd.append('--use-orthogonal')
            
            # FTLE method selection
            if use_variational_ftle:
                cmd.append('--ftle-variational')
            else:
                cmd.append('--ftle-finite-diff')
            
            cmd.extend(extra_params)
            
            print(f"  > [Local] Processing t={t_val}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Local Metric calculation at t={t_val}")
                sys.exit(e.returncode)
    else:
        print("  > [Local] Skipped (disabled in config).")

    # =========================================================
    # PART B: Global Metrics (Fresh Integration)
    # =========================================================
    has_global = (ac.get('compute_global_stod_forward', False) or 
                 ac.get('compute_global_stod_backward', False) or
                 ac.get('compute_global_finstod_forward', False) or 
                 ac.get('compute_global_finstod_backward', False))
    
    if has_global:
        # For global, we use a dummy input file (trajectories are computed fresh)
        dummy_input = os.path.join(out_base, "trajectory_cache", "dummy.h5")
        if not os.path.exists(dummy_input):
            print(f"ERROR: Dummy input file not found at {dummy_input}")
            sys.exit(1)
        
        for t_snap in snaps:
            t_val = float(t_snap)
            out_name = f"results_chunk_{args.worker_id:04d}_global_snap_{t_val:.2f}.h5"
            out_path = os.path.join(res_dir, out_name)
            
            cmd = [
                binary,
                sys_name,
                dummy_input,  # Use dummy file for global (fresh integration)
                out_path,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                str(t_val),
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row)
            ]
            
            # Add analysis control flags for global metrics
            if ac.get('compute_global_stod_forward', False):
                cmd.append('--compute-stod-forward')
            if ac.get('compute_global_stod_backward', False):
                cmd.append('--compute-stod-backward')
            if ac.get('compute_global_finstod_forward', False):
                cmd.append('--compute-finstod-forward')
            if ac.get('compute_global_finstod_backward', False):
                cmd.append('--compute-finstod-backward')
            
            # Orthogonal field mode
            if use_orthogonal:
                cmd.append('--use-orthogonal')
            
            # FTLE method selection (for any global FTLE if added in future)
            if use_variational_ftle:
                cmd.append('--ftle-variational')
            else:
                cmd.append('--ftle-finite-diff')
            
            cmd.extend(extra_params)
            
            print(f"  > [Global] Processing t={t_val}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Global Metric calculation at t={t_val}")
                sys.exit(e.returncode)
    else:
        print("  > [Global] Skipped (disabled in config).")

    # Note: LB and OLB calculations have been moved to a separate stage (Stage 2B)
    # to improve load balancing. They are now computed by dedicated workers
    # that run after snapshot metrics complete.
    print("  > [LB/OLB] Skipped (moved to separate Stage 2B for better load balancing).")

    print(f"--- Worker {args.worker_id} Finished ---")

if __name__ == "__main__":
    main()



