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
    parser.add_argument("--stage", type=str, required=True, 
                       choices=['ftle', 'local_stod', 'global_stod', 'lb', 'olb'])
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

    # Input file (Chunk from GenTraj)
    input_h5 = os.path.join(traj_folder, f"chunk_{args.worker_id:04d}.h5")
    
    if not os.path.exists(input_h5):
        # Graceful exit if job array is larger than rows
        print(f"Worker {args.worker_id}: Input {input_h5} not found (likely empty row range). Exiting.")
        sys.exit(0)

    # 3. Calculate Row Range
    gp = config['grid_params']
    ny = int(gp['grid_resolution_y'])
    
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
    ac = config.get('analysis_controls', {})

    print(f"--- Worker {args.worker_id} Stage: {args.stage} (Rows {start_row}-{end_row}) ---")

    # =========================================================
    # STAGE 1: FTLE Only (Lightest)
    # =========================================================
    if args.stage == 'ftle':
        has_ftle = (ac.get('compute_ftle_forward', False) or 
                   ac.get('compute_ftle_backward', False))
        
        if not has_ftle:
            print("  > [FTLE] Skipped (disabled in config).")
            sys.exit(0)

        for t_snap in snaps:
            t_val = float(t_snap)
            out_name = f"results_chunk_{args.worker_id:04d}_ftle_snap_{t_val:.2f}.h5"
            out_path = os.path.join(res_dir, out_name)
            
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
                str(end_row),
                "--ftle-only"  # Flag to compute only FTLE
            ]
            cmd.extend(extra_params)
            
            print(f"  > [FTLE] Processing t={t_val}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in FTLE calculation at t={t_val}")
                sys.exit(e.returncode)

    # =========================================================
    # STAGE 2: Local STOD (Using Pre-computed Trajectories)
    # =========================================================
    elif args.stage == 'local_stod':
        has_local = (ac.get('compute_local_stod_forward', False) or 
                    ac.get('compute_local_stod_backward', False) or
                    ac.get('compute_local_finstod_forward', False) or 
                    ac.get('compute_local_finstod_backward', False))
        
        if not has_local:
            print("  > [Local STOD] Skipped (disabled in config).")
            sys.exit(0)

        for t_snap in snaps:
            t_val = float(t_snap)
            out_name = f"results_chunk_{args.worker_id:04d}_local_snap_{t_val:.2f}.h5"
            out_path = os.path.join(res_dir, out_name)
            
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
                str(end_row),
                "--local-stod"  # Flag to use pre-computed trajectories
            ]
            cmd.extend(extra_params)
            
            print(f"  > [Local STOD] Processing t={t_val}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Local STOD calculation at t={t_val}")
                sys.exit(e.returncode)

    # =========================================================
    # STAGE 3: Global STOD (Fresh Integration)
    # =========================================================
    elif args.stage == 'global_stod':
        has_global = (ac.get('compute_global_stod_forward', False) or 
                     ac.get('compute_global_stod_backward', False) or
                     ac.get('compute_global_finstod_forward', False) or 
                     ac.get('compute_global_finstod_backward', False))
        
        if not has_global:
            print("  > [Global STOD] Skipped (disabled in config).")
            sys.exit(0)

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
                str(end_row),
                "--global-stod"  # Flag for fresh integration
            ]
            cmd.extend(extra_params)
            
            print(f"  > [Global STOD] Processing t={t_val}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in Global STOD calculation at t={t_val}")
                sys.exit(e.returncode)

    # =========================================================
    # STAGE 4: LB (Lagrangian Betweenness)
    # =========================================================
    elif args.stage == 'lb':
        do_lb = ac.get('compute_lb', False)
        lb_slices = ac.get('lb_slices', 300)
        
        if not do_lb:
            print("  > [LB] Skipped (disabled in config).")
            sys.exit(0)

        print(f"  > [LB] Processing Integral (N={lb_slices})...")
        out_name_lb = f"results_chunk_{args.worker_id:04d}_exact_lb.h5"
        out_path_lb = os.path.join(res_dir, out_name_lb)

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
            print(f"!!! Error in LB calculation")
            sys.exit(e.returncode)

    # =========================================================
    # STAGE 5: OLB (Optimal Lagrangian Betweenness)
    # =========================================================
    elif args.stage == 'olb':
        do_olb_local_stod = ac.get('compute_olb_local_stod', False)
        do_olb_local_finstod = ac.get('compute_olb_local_finstod', False)
        do_olb_global_stod = ac.get('compute_olb_global_stod', False)
        do_olb_global_finstod = ac.get('compute_olb_global_finstod', False)
        olb_slices = ac.get('olb_slices', 300)

        if not (do_olb_local_stod or do_olb_local_finstod or 
                do_olb_global_stod or do_olb_global_finstod):
            print("  > [OLB] Skipped (disabled in config).")
            sys.exit(0)

        if do_olb_local_stod:
            print(f"  > [OLB Local STOD] Processing Integral (N={olb_slices})...")
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
                print(f"!!! Error in OLB Local STOD calculation")
                sys.exit(e.returncode)

        if do_olb_local_finstod:
            print(f"  > [OLB Local FINSTOD] Processing Integral (N={olb_slices})...")
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
                print(f"!!! Error in OLB Local FINSTOD calculation")
                sys.exit(e.returncode)

        if do_olb_global_stod:
            print(f"  > [OLB Global STOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_global_stod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            dummy_input = os.path.join(out_base, "trajectory_cache", "dummy.h5")
            cmd_olb = [
                binary,
                sys_name,
                dummy_input,
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-stod",
                "--global-stod",
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in OLB Global STOD calculation")
                sys.exit(e.returncode)

        if do_olb_global_finstod:
            print(f"  > [OLB Global FINSTOD] Processing Integral (N={olb_slices})...")
            out_name_olb = f"results_chunk_{args.worker_id:04d}_exact_olb_global_finstod.h5"
            out_path_olb = os.path.join(res_dir, out_name_olb)

            dummy_input = os.path.join(out_base, "trajectory_cache", "dummy.h5")
            cmd_olb = [
                binary,
                sys_name,
                dummy_input,
                out_path_olb,
                str(gp['grid_resolution_x']), str(gp['grid_resolution_y']),
                str(gp['x_min']), str(gp['x_max']),
                str(gp['y_min']), str(gp['y_max']),
                "0.0", 
                str(tp['total_integration_time_end']),
                str(start_row),
                str(end_row),
                "--compute-olb-finstod",
                "--global-stod",
                f"olb_slices={olb_slices}"
            ]
            cmd_olb.extend(extra_params)

            try:
                subprocess.run(cmd_olb, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Error in OLB Global FINSTOD calculation")
                sys.exit(e.returncode)

    print(f"--- Worker {args.worker_id} Stage {args.stage} Finished ---")

if __name__ == "__main__":
    main()



