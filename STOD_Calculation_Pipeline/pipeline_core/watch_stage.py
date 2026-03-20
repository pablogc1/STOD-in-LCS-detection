import os
import sys
import time
import glob
import argparse
import subprocess
from tqdm import tqdm

def is_job_running(job_id):
    """Checks if the job is still in the Slurm queue."""
    try:
        # squeue -h (no header) -j <job_id>
        res = subprocess.run(['squeue', '-h', '-j', str(job_id)], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return len(res.stdout.strip()) > 0
    except Exception:
        return True

def count_files(pattern):
    """Count files matching pattern(s). Supports space-separated multiple patterns."""
    patterns = pattern.split()
    total = 0
    for p in patterns:
        total += len(glob.glob(p))
    return total

def format_time(seconds):
    """Formats seconds to HH:MM:SS"""
    if seconds is None: return "--:--:--"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True, help="Slurm Job ID to watch")
    parser.add_argument("--pattern", required=True, help="File pattern to count")
    parser.add_argument("--total", type=int, required=True, help="Expected number of files")
    parser.add_argument("--name", default="Stage", help="Name of the stage for display")
    args = parser.parse_args()

    # Detect if interactive
    interactive = sys.stdout.isatty()
    
    current = 0
    start_time = time.time()
    
    # Setup TQDM if interactive
    pbar = None
    if interactive:
        pbar = tqdm(total=args.total, desc=f"Wait: {args.name}", unit="file")
    else:
        # Log start
        print(f"[STATUS] Starting watch for {args.name} (Total: {args.total} files)")
        sys.stdout.flush()

    while True:
        # 1. Update Count
        new_count = count_files(args.pattern)
        
        # Update interactive bar
        if interactive and pbar:
            diff = new_count - current
            if diff > 0:
                pbar.update(diff)
        
        current = new_count
        elapsed = time.time() - start_time
        
        # 2. Check Success Condition
        if current >= args.total:
            if interactive and pbar: pbar.close()
            print(f"\n[SUCCESS] {args.name}: All {args.total} files generated in {format_time(elapsed)}.")
            break

        # 3. Check Job Status
        if not is_job_running(args.job_id):
            # Job finished - wait a bit for final files to be written
            time.sleep(10)
            final_check = count_files(args.pattern)
            if final_check >= args.total:
                if interactive and pbar: pbar.close()
                print(f"\n[SUCCESS] {args.name}: Job ended and files are present. Time: {format_time(elapsed)}.")
                break
            else:
                if interactive and pbar: pbar.close()
                print(f"\n[FAILURE] {args.name}: Job {args.job_id} disappeared but only {final_check}/{args.total} files found.")
                print(f"  Pattern(s): {args.pattern}")
                print(f"  Check SLURM logs for job {args.job_id} to diagnose failures.")
                sys.exit(1)

        # 4. Non-interactive Log update (every ~30s)
        if not interactive:
            # Calculate Rate and ETA
            if current > 0:
                rate = current / elapsed
                remaining = args.total - current
                eta_seconds = remaining / rate
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "--:--:--"

            pct = (current / args.total) * 100.0
            print(f"[STATUS] {args.name}: {current}/{args.total} ({pct:.1f}%) | Elapsed: {format_time(elapsed)} | ETA: {eta_str}")
            sys.stdout.flush()

        time.sleep(5 if interactive else 30)

if __name__ == "__main__":
    main()

