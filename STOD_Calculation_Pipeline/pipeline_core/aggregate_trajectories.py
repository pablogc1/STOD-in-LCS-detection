# -*- coding: utf-8 -*-
"""
Aggregate Trajectories (direction-aware + row index)
====================================================
- Merges partial trajectory pickles by direction (only those enabled).
- Produces direction-specific manifests listing the partial files.
- Emits a row_index.json mapping each partial pickle to its [start_row, end_row).
"""

import os
import sys
import yaml
import pickle
import glob
import json
from math import ceil
from tqdm import tqdm

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _get_flags(cfg: dict):
    ac = cfg.get("analysis_controls", {}) or {}
    enable_forward = ac.get("enable_forward", True)
    enable_backward = ac.get("enable_backward", True)
    if not enable_forward and not enable_backward:
        enable_forward = True
    return bool(enable_forward), bool(enable_backward)

def aggregate_pickles(pattern, output_file):
    """Merge all pickle dicts matching pattern into one."""
    merged = {}
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[Info] No partial files found for {os.path.basename(pattern)} - skipping merge.")
        return merged, files

    for fpath in tqdm(files, desc=f"Aggregating {os.path.basename(pattern)}"):
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                merged.update(data)
        except Exception as e:
            print(f"  -> Skipped {fpath}: {e}")

    with open(output_file, "wb") as f:
        pickle.dump(merged, f)

    print(f"  -> Saved merged file: {output_file} ({len(merged)} entries)")
    return merged, files

def write_manifest(files, manifest_path):
    with open(manifest_path, "w") as f:
        for p in files:
            f.write(os.path.abspath(p) + "\n")
    print(f"  -> Manifest written: {manifest_path} ({len(files)} files)")

def _infer_row_ranges(grid_res_y: int, partial_paths: list):
    """
    Infer [start_row, end_row) for each partial_fwd_cache_job_{id}.pkl / bwd...
    We assume Stage 1 split rows into contiguous bands with:
      rows_per_job = ceil(grid_res_y / num_partials)
      start = (job_id-1) * rows_per_job
      end   = min(grid_res_y, start + rows_per_job)
    """
    # Extract numeric job_id from filenames of the form *_job_{ID}.pkl
    def _job_id(path: str):
        base = os.path.basename(path)
        # Split by '_job_' and strip extension
        if "_job_" in base:
            try:
                return int(base.split("_job_")[1].split(".")[0])
            except Exception:
                return None
        return None

    ids = [jid for jid in (_job_id(p) for p in partial_paths) if jid is not None]
    if not ids:
        return {}

    num_partials = len(ids)
    rows_per_job = int(ceil(grid_res_y / num_partials))
    row_index = {}

    for p in partial_paths:
        jid = _job_id(p)
        if jid is None:
            continue
        start_row = (jid - 1) * rows_per_job
        end_row = min(grid_res_y, start_row + rows_per_job)
        row_index[os.path.abspath(p)] = {"start_row": start_row, "end_row": end_row}
    return row_index

def main():
    print("--- Starting Generic Trajectory Aggregation Job ---")
    config_path = os.environ.get("SYSTEM_CONFIG_FILE")
    if not config_path:
        sys.exit("SYSTEM_CONFIG_FILE not set; cannot continue.")
    print(f"System Config File: {config_path}")

    config = load_config(config_path)
    enable_forward, enable_backward = _get_flags(config)

    out_dir = config["output_base_folder"]
    partial_dir = os.path.join(out_dir, "trajectory_cache", "partial")
    traj_dir = os.path.join(out_dir, "trajectory_cache")
    merged_dir = os.path.join(traj_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # Patterns
    pattern_fwd = os.path.join(partial_dir, "partial_fwd_cache_job_*.pkl")
    pattern_bwd = os.path.join(partial_dir, "partial_bwd_cache_job_*.pkl")

    # Outputs
    out_fwd = os.path.join(merged_dir, "merged_fwd_cache.pkl")
    out_bwd = os.path.join(merged_dir, "merged_bwd_cache.pkl")

    # Manifests
    fwd_manifest = os.path.join(traj_dir, "fwd_traj_manifest.txt")
    bwd_manifest = os.path.join(traj_dir, "bwd_traj_manifest.txt")

    # Forward (if enabled)
    fwd_files = []
    if enable_forward:
        print("\nMerging forward caches...")
        _, fwd_files = aggregate_pickles(pattern_fwd, out_fwd)
        write_manifest(fwd_files, fwd_manifest)
    else:
        for p in (out_fwd, fwd_manifest):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        print("\nForward disabled — skipped merging and removed any stale forward artifacts.")

    # Backward (if enabled)
    bwd_files = []
    if enable_backward:
        print("\nMerging backward caches...")
        _, bwd_files = aggregate_pickles(pattern_bwd, out_bwd)
        write_manifest(bwd_files, bwd_manifest)
    else:
        for p in (out_bwd, bwd_manifest):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        print("\nBackward disabled — skipped merging and removed any stale backward artifacts.")

    # -------- NEW: emit row_index.json so workers can band-load only what they need
    grid_res_y = config["grid_params"]["grid_resolution_y"]
    row_index = {}
    if fwd_files:
        row_index.update(_infer_row_ranges(grid_res_y, fwd_files))
    if bwd_files:
        row_index.update(_infer_row_ranges(grid_res_y, bwd_files))

    index_path = os.path.join(traj_dir, "row_index.json")
    with open(index_path, "w") as jf:
        json.dump(row_index, jf, indent=2)
    print(f"\nRow index written: {index_path} (entries: {len(row_index)})")

    print("\n--- Aggregation complete ---")

if __name__ == "__main__":
    main()

