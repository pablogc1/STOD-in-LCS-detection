# ----------------------------------------------------------------------
#        PIPELINE CORE: STAGE 3 - Generic Master Task List Generation
# ----------------------------------------------------------------------

import os
import sys
import yaml

def main():
    print("--- GENERATING MASTER TASK LIST ---")
    config_path = os.environ.get('SYSTEM_CONFIG_FILE')
    if not config_path or not os.path.exists(config_path):
        print("ERROR: SYSTEM_CONFIG_FILE is not set or invalid.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    snapshot_times = cfg['time_params']['snapshot_times']
    grid_y_res = cfg['grid_params']['grid_resolution_y']
    chunk = cfg['hpc_params']['task_chunk_size']
    out_base = cfg['output_base_folder']
    list_name = cfg['master_task_list_filename']
    list_path = os.path.join(out_base, list_name)

    os.makedirs(out_base, exist_ok=True)

    total = 0
    with open(list_path, 'w') as f:
        for t in snapshot_times:
            for start_row in range(0, grid_y_res, chunk):
                end_row = min(start_row + chunk, grid_y_res)
                f.write(f"{t} {start_row} {end_row}\n")
                total += 1

    print(f"System Config File: {config_path}")
    print(f"  -> Master task list written to {list_path}")
    print(f"  -> Total tasks: {total}")
    print("\n--- Master Task List Generation Complete ---")

if __name__ == "__main__":
    main()

