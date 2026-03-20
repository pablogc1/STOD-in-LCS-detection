#!/usr/bin/env python3
"""
Aggregate timing data from worker logs and display a summary.
Parses [TIMING_DATA] lines from worker output files.

Timing breakdown:
- Standard metrics (FTLE, FLI, LD): each includes its own trajectory integration
- STOD metrics: broken down into trajectory generation, STOD scoring, FINSTOD scoring
"""

import os
import sys
import glob
import re
import argparse

def parse_timing_line(line):
    """Parse a [TIMING_DATA] line and return a dict of values."""
    pattern = r'\[TIMING_DATA\]\s*(.*)'
    match = re.match(pattern, line)
    if not match:
        return None
    
    data = {}
    parts = match.group(1).split()
    for part in parts:
        if '=' in part:
            key, val = part.split('=', 1)
            try:
                data[key] = float(val)
            except ValueError:
                pass
    return data

def aggregate_timing(log_folder, job_id_prefix="worker_snaps"):
    """
    Aggregate timing data from all worker log files.
    Returns a dict with summed timing values across all workers.
    """
    pattern = os.path.join(log_folder, f"{job_id_prefix}_*.out")
    log_files = glob.glob(pattern)
    
    totals = {
        # Standard metrics
        'FTLE_FWD': 0.0, 'FTLE_BWD': 0.0,
        'FLI_FWD': 0.0, 'FLI_BWD': 0.0,
        'LD_FWD': 0.0, 'LD_BWD': 0.0,
        # STOD breakdown
        'STOD_TRAJ_FWD': 0.0, 'STOD_TRAJ_BWD': 0.0,
        'STOD_FWD': 0.0, 'STOD_BWD': 0.0,
        'FINSTOD_FWD': 0.0, 'FINSTOD_BWD': 0.0,
        # Totals
        'TOTAL_STANDARD': 0.0, 
        'TOTAL_STOD_TRAJ': 0.0,
        'TOTAL_STOD': 0.0,
        'TOTAL_FINSTOD': 0.0,
        'TOTAL_STOD': 0.0,
        'TOTAL': 0.0
    }
    
    worker_count = 0
    snapshot_count = 0
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if '[TIMING_DATA]' in line:
                        data = parse_timing_line(line.strip())
                        if data:
                            snapshot_count += 1
                            for key in totals:
                                if key in data:
                                    totals[key] += data[key]
            worker_count += 1
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}", file=sys.stderr)
    
    return totals, worker_count, snapshot_count

def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_summary(totals, worker_count, snapshot_count):
    """Print a formatted timing summary."""
    # Calculate totals from components
    total_ftle = totals['FTLE_FWD'] + totals['FTLE_BWD']
    total_fli = totals['FLI_FWD'] + totals['FLI_BWD']
    total_ld = totals['LD_FWD'] + totals['LD_BWD']
    total_standard = total_ftle + total_fli + total_ld
    
    total_stod_traj = totals['STOD_TRAJ_FWD'] + totals['STOD_TRAJ_BWD']
    total_stod = totals['STOD_FWD'] + totals['STOD_BWD']
    total_finstod = totals['FINSTOD_FWD'] + totals['FINSTOD_BWD']
    total_stod = total_stod_traj + total_stod + total_finstod
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║              STAGE 2A TIMING SUMMARY (Aggregated from all workers)           ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Workers: {worker_count:4d}  |  Snapshots processed: {snapshot_count:4d}                              ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  STANDARD METRICS (each includes trajectory integration):                    ║")
    if total_ftle > 0:
        print(f"║    FTLE:       {format_time(total_ftle):>10s}  ({total_ftle:10.1f}s cumulative CPU time)           ║")
    if total_fli > 0:
        print(f"║    FLI:        {format_time(total_fli):>10s}  ({total_fli:10.1f}s cumulative CPU time)           ║")
    if total_ld > 0:
        print(f"║    LD:         {format_time(total_ld):>10s}  ({total_ld:10.1f}s cumulative CPU time)           ║")
    print(f"║    ─────────────────────────────────────────────────────────────────────    ║")
    print(f"║    Standard Total: {format_time(total_standard):>10s}  ({total_standard:10.1f}s)                          ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  STOD METRICS (broken down by phase):                                        ║")
    if total_stod_traj > 0:
        print(f"║    Traj Gen:   {format_time(total_stod_traj):>10s}  ({total_stod_traj:10.1f}s cumulative CPU time)           ║")
    if total_stod > 0:
        print(f"║    STOD:     {format_time(total_stod):>10s}  ({total_stod:10.1f}s cumulative CPU time)           ║")
    if total_finstod > 0:
        print(f"║    FINSTOD:    {format_time(total_finstod):>10s}  ({total_finstod:10.1f}s cumulative CPU time)           ║")
    print(f"║    ─────────────────────────────────────────────────────────────────────    ║")
    print(f"║    STOD Total:     {format_time(total_stod):>10s}  ({total_stod:10.1f}s)                          ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  COMPARISON:                                                                 ║")
    
    # STOD vs Standard comparison
    if total_standard > 0 and total_stod > 0:
        ratio = total_stod / total_standard
        if ratio > 1:
            comparison = f"STOD total is {(ratio - 1) * 100:.1f}% slower than standard"
        elif ratio < 1:
            comparison = f"STOD total is {(1 - ratio) * 100:.1f}% faster than standard"
        else:
            comparison = "STOD and standard take about the same time"
        print(f"║    STOD / Standard = {ratio:.2f}x                                                 ║")
        print(f"║      ({comparison:66s})║")
    
    # FINSTOD vs STOD comparison
    if total_stod > 0 and total_finstod > 0:
        ratio_if = total_finstod / total_stod
        print(f"║    FINSTOD / STOD = {ratio_if:.2f}x                                                ║")
    
    # Traj Gen percentage of total STOD
    if total_stod > 0 and total_stod_traj > 0:
        traj_pct = (total_stod_traj / total_stod) * 100
        print(f"║    Trajectory generation is {traj_pct:5.1f}% of total STOD time                     ║")
    
    # Per-snapshot averages
    if snapshot_count > 0:
        print("╠══════════════════════════════════════════════════════════════════════════════╣")
        print("║  PER-SNAPSHOT AVERAGES:                                                      ║")
        if total_standard > 0:
            avg_std = total_standard / snapshot_count
            print(f"║    Standard metrics: {avg_std:7.2f}s per snapshot                                ║")
        if total_stod > 0:
            avg_stod = total_stod / snapshot_count
            print(f"║    STOD metrics:     {avg_stod:7.2f}s per snapshot                                ║")
        if total_stod > 0:
            avg_stod = total_stod / snapshot_count
            print(f"║      - STOD only:  {avg_stod:7.2f}s per snapshot                                ║")
        if total_finstod > 0:
            avg_finstod = total_finstod / snapshot_count
            print(f"║      - FINSTOD only: {avg_finstod:7.2f}s per snapshot                                ║")
    
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

def main():
    parser = argparse.ArgumentParser(description="Aggregate timing data from worker logs")
    parser.add_argument("--log_folder", required=True, help="Path to SLURM log folder")
    parser.add_argument("--job_prefix", default="worker_snaps", help="Log file prefix (default: worker_snaps)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.log_folder):
        print(f"Error: Log folder not found: {args.log_folder}", file=sys.stderr)
        sys.exit(1)
    
    totals, worker_count, snapshot_count = aggregate_timing(args.log_folder, args.job_prefix)
    
    if worker_count == 0:
        print("No worker logs found.")
        sys.exit(0)
    
    print_summary(totals, worker_count, snapshot_count)

if __name__ == "__main__":
    main()
