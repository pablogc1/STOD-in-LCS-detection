import os
import sys
import argparse
import yaml
import subprocess
import time

def get_system_shortname(module_str):
    if "." in module_str:
        return module_str.split(".")[-1]
    return module_str

def collect_numeric_params(config):
    params = []
    sections = ['physics_params', 'grid_params', 'time_params']
    for sec in sections:
        if sec in config and isinstance(config[sec], dict):
            for key, val in config[sec].items():
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    params.append(f"{key}={val}")
    return params

def main():
    parser = argparse.ArgumentParser(description="Generic C++ Wrapper for Trajectory Generation")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_cores", type=int, required=True)
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = os.environ.get('SYSTEM_CONFIG_FILE')
    if not config_path or not os.path.exists(config_path):
        print("ERROR: SYSTEM_CONFIG_FILE not set.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Determine if forward/backward trajectories are needed ---
    controls = config.get("analysis_controls", {})
    
    # Check if any backward metrics are enabled
    has_backward_metrics = (
        controls.get("compute_ftle_backward", False) or
        controls.get("compute_local_stod_backward", False) or
        controls.get("compute_local_finstod_backward", False) or
        controls.get("compute_global_stod_backward", False) or
        controls.get("compute_global_finstod_backward", False) or
        controls.get("compute_fli_backward", False) or
        controls.get("compute_ld_backward", False) or
        controls.get("compute_lb", False) or  # LB uses backward FTLE
        controls.get("compute_olb_local_stod", False) or
        controls.get("compute_olb_local_finstod", False) or
        controls.get("compute_olb_global_stod", False) or
        controls.get("compute_olb_global_finstod", False)
    )
    
    # Check if any forward metrics are enabled (default to True if no controls specified)
    has_forward_metrics = (
        controls.get("compute_ftle_forward", True) or
        controls.get("compute_local_stod_forward", False) or
        controls.get("compute_local_finstod_forward", False) or
        controls.get("compute_global_stod_forward", False) or
        controls.get("compute_global_finstod_forward", False) or
        controls.get("compute_fli_forward", False) or
        controls.get("compute_ld_forward", False) or
        controls.get("compute_lb", False) or  # LB uses forward FTLE
        controls.get("compute_olb_local_stod", False) or
        controls.get("compute_olb_local_finstod", False) or
        controls.get("compute_olb_global_stod", False) or
        controls.get("compute_olb_global_finstod", False)
    )
    
    # Allow explicit override via enable_forward/enable_backward if present
    if "enable_forward" in controls:
        has_forward_metrics = controls.get("enable_forward", True)
    if "enable_backward" in controls:
        has_backward_metrics = controls.get("enable_backward", False)
    
    do_fwd = 1 if has_forward_metrics else 0
    do_bwd = 1 if has_backward_metrics else 0

    if do_fwd == 0 and do_bwd == 0:
        print("WARNING: Both forward and backward analysis disabled. Enabling Forward by default.")
        do_fwd = 1
    
    # --- Read Orthogonal Mode ---
    # Check for environment variable override (used by "both" mode in master_run.sh)
    field_mode_override = os.environ.get("FIELD_MODE_OVERRIDE")
    if field_mode_override:
        field_mode = field_mode_override
        print(f"[INFO] Using FIELD_MODE_OVERRIDE: {field_mode}")
    else:
        field_mode = config.get("field_mode", "standard")
    
    use_orthogonal = 1 if field_mode == "orthogonal" else 0
    
    # Warning for truly high-dimensional systems with orthogonal mode
    # Note: Lorenz is now a 2D Poincaré section (z fixed), so it's fine
    # Only LiCN is a true 4D system where orthogonal mode is problematic
    sys_name_check = get_system_shortname(config['system_module']).lower()
    if use_orthogonal and sys_name_check in ['licn']:
        print("")
        print("=" * 70)
        print("[WARNING] ORTHOGONAL MODE ON HIGH-DIMENSIONAL SYSTEM")
        print(f"  System: {sys_name_check}")
        print("  The orthogonal field for 4D systems creates non-physical dynamics.")
        print("  Only the first two velocity components are rotated.")
        print("  This may cause:")
        print("    - Very slow integration (numerical instability)")
        print("    - Non-physical trajectories")
        print("    - Meaningless metric values")
        print("  Results should be interpreted with extreme caution.")
        print("=" * 70)
        print("")

    # 2. Extract Standard Grid/Time Info
    sys_name = get_system_shortname(config['system_module'])
    gp = config['grid_params']
    tp = config['time_params']

    rows_per_job = (gp['grid_resolution_y'] + args.total_jobs - 1) // args.total_jobs
    start_row = (args.job_id - 1) * rows_per_job
    end_row = min(gp['grid_resolution_y'], start_row + rows_per_job)

    if start_row >= end_row:
        print(f"Job {args.job_id} has no rows to process. Exiting.")
        sys.exit(0)

    # 3. Prepare Paths
    output_base = config['output_base_folder']
    traj_folder = os.path.join(output_base, "trajectory_cache", "partial")
    os.makedirs(traj_folder, exist_ok=True)
    output_filename = os.path.join(traj_folder, f"chunk_{args.job_id:04d}.h5")

    # 4. Build Command
    binary_path = "./cpp_backend/gen_traj"
    if not os.path.exists(binary_path):
        binary_path = os.path.join(os.getcwd(), "cpp_backend", "gen_traj")

    cmd = [
        binary_path,
        sys_name,
        str(gp['grid_resolution_x']),
        str(gp['grid_resolution_y']),
        str(gp['x_min']), str(gp['x_max']),
        str(gp['y_min']), str(gp['y_max']),
        str(tp['total_integration_time_end']),
        str(tp['time_step_dt']),
        str(start_row),
        str(end_row),
        output_filename,
        str(do_fwd),         # <--- Passing Flag
        str(do_bwd),         # <--- Passing Flag
        str(use_orthogonal)  # <--- Passing Orthogonal Flag
    ]

    extra_params = collect_numeric_params(config)
    cmd.extend(extra_params)
    
    if extra_params:
        print("--- Passing Detected Parameters to C++ ---")
        for p in extra_params:
            print(f"  -> {p}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.num_cores)

    print(f"--- Executing: {sys_name} (Fwd={do_fwd}, Bwd={do_bwd}, Orthogonal={use_orthogonal}) ---")
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"C++ Error Code: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    print(f"Done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()


