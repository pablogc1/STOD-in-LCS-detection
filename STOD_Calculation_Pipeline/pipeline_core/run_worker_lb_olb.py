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
    traj_folder = os.path.join(out_base, "trajectory_cache", "partial")
    res_dir = os.path.join(out_base, "partial_results")
    os.makedirs(res_dir, exist_ok=True)

    # Input file (Chunk from GenTraj) - same as regular workers
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
    extra_params = collect_numeric_params(config)

    print(f"--- Worker {args.worker_id} (LB/OLB Only, Rows {start_row}-{end_row}) ---")

    ac = config.get('analysis_controls', {})

    # =========================================================
    # PART A: Exact Lagrangian Betweenness (Optional)
    # =========================================================
    do_exact_lb = ac.get('compute_lb', False)
    lb_slices = ac.get('lb_slices', 300)

    if do_exact_lb:
        print(f"  > [Exact LB] Processing Integral (N={lb_slices})...")
        out_name_lb = f"results_chunk_{args.worker_id:04d}_exact_lb.h5"
        out_path_lb = os.path.join(res_dir, out_name_lb)

        # We pass t_snap=0.0 as dummy, and append LB flags
        cmd_lb = [
            binary,
            sys_name,
            input_h5,
            out_path_lb,
            str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
            str(gp['x_min']), str(gp['x_max']),
            str(gp['y_min']), str(gp['y_max']),
            "0.0", 
            str(tp['total_integration_time_end']),
            str(start_row),
            str(end_row),
            "--compute-lb",
            f"lb_slices={lb_slices}"
        ]
        cmd_lb.extend(extra_params)

        try:
            subprocess.run(cmd_lb, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Error in Exact LB calculation")
            sys.exit(e.returncode)
    else:
        print("  > [Exact LB] Skipped (disabled in config).")

    # =========================================================
    # PART B: Exact OLB (Optional) - LB using STOD
    # =========================================================
    # OLB is computed on-the-fly at each time slice (matching LB approach)
    do_olb_local_stod = ac.get('compute_olb_local_stod', False)
    do_olb_local_finstod = ac.get('compute_olb_local_finstod', False)
    do_olb_global_stod = ac.get('compute_olb_global_stod', False)
    do_olb_global_finstod = ac.get('compute_olb_global_finstod', False)
    olb_slices = ac.get('olb_slices', 300)

    if do_olb_local_stod or do_olb_local_finstod or do_olb_global_stod or do_olb_global_finstod:
        if do_olb_local_stod:
            print(f"  > [Exact OLB Local STOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_local_stod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            cmd_olb = [
                binary,
                sys_name,
                input_h5,
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-stod",
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Exact OLB Local STOD calculation")
                sys.exit(e.returncode)

        if do_olb_local_finstod:
            print(f"  > [Exact OLB Local FINSTOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_local_finstod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            cmd_olb = [
                binary,
                sys_name,
                input_h5,
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-finstod",
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Exact OLB Local FINSTOD calculation")
                sys.exit(e.returncode)

        # Global OLB calculations
        dummy_input = os.path.join(out_base, "trajectory_cache", "dummy.h5")
        if not os.path.exists(dummy_input):
            print(f"ERROR: Dummy input file not found at {dummy_input}")
            sys.exit(1)

        if do_olb_global_stod:
            print(f"  > [Exact OLB Global STOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_global_stod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            cmd_olb = [
                binary,
                sys_name,
                dummy_input,  # Use dummy file for global
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-stod",
                "--global-stod",  # Flag to enable global STOD computation
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Exact OLB Global STOD calculation")
                sys.exit(e.returncode)

        if do_olb_global_finstod:
            print(f"  > [Exact OLB Global FINSTOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_global_finstod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            cmd_olb = [
                binary,
                sys_name,
                dummy_input,  # Use dummy file for global
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-finstod",
                "--global-stod",  # Flag to enable global STOD computation
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Exact OLB Global FINSTOD calculation")
                sys.exit(e.returncode)
    else:
        print("  > [Exact OLB] Skipped (disabled in config).")

    print(f"--- Worker {args.worker_id} (LB/OLB) Finished ---")

if __name__ == "__main__":
    main()



