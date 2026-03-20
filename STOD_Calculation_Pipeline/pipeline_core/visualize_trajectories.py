#!/usr/bin/env python3
"""
Trajectory Evolution Visualizer - HPC Optimized
===============================================
Memory-efficient version designed for SLURM Job Arrays.
Each task generates a single frame, allowing for massive parallelization.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import importlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_core.integrator import generate_trajectory

def load_system(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"ERROR: Could not import system module '{module_name}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize Trajectory Evolution (HPC)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trajectory_evolution")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--n_snapshots", type=int, default=15)
    parser.add_argument("--total_time", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--frame_idx", type=int)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--max_points", type=int, default=100, help="Max points per line to save memory")

    args = parser.parse_args()

    # Detect SLURM Array Task ID
    slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_idx is not None and args.frame_idx is None:
        args.frame_idx = int(slurm_idx)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    system = load_system(config['system_module'])
    physics_params = config.get('physics_params', {})
    gp = config['grid_params']
    tp = config['time_params']

    total_time = args.total_time if args.total_time else tp['total_integration_time_end']
    dt = args.dt if args.dt else tp.get('time_step_dt', 0.1)
    
    snapshot_times = np.linspace(0, total_time, args.n_snapshots)
    
    if args.frame_idx is not None:
        target_times = [snapshot_times[args.frame_idx]]
        indices = [args.frame_idx]
    else:
        target_times = snapshot_times
        indices = range(len(snapshot_times))

    # Setup Grid
    x_lines = np.linspace(gp['x_min'], gp['x_max'], args.grid_size + 1)
    y_lines = np.linspace(gp['y_min'], gp['y_max'], args.grid_size + 1)
    x_centers = (x_lines[:-1] + x_lines[1:]) / 2
    y_centers = (y_lines[:-1] + y_lines[1:]) / 2
    X_init, Y_init = np.meshgrid(x_centers, y_centers)
    X0, Y0 = X_init.ravel(), Y_init.ravel()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for f_idx, t_snap in zip(indices, target_times):
        print(f"--- Generating Frame {f_idx} (t={t_snap:.2f}) ---")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw Background Grid
        for x in x_lines: ax.axvline(x, color='gray', lw=0.4, alpha=0.1)
        for y in y_lines: ax.axhline(y, color='gray', lw=0.4, alpha=0.1)

        # Batch integration and plotting
        for i in range(0, len(X0), args.batch_size):
            end_i = min(i + args.batch_size, len(X0))
            batch_X0 = X0[i:end_i]
            batch_Y0 = Y0[i:end_i]
            
            paths = []
            for x0, y0 in zip(batch_X0, batch_Y0):
                if t_snap == 0:
                    paths.append(np.array([[x0, y0], [x0, y0]]))
                    continue
                
                traj = generate_trajectory(system.get_velocity_field, [x0, y0], t_snap, dt, 1, physics_params)
                
                # CRITICAL: Subsample to max_points (e.g. 100) to match Spyder logic and save memory
                step = max(1, len(traj) // args.max_points)
                paths.append(traj[::step])

            lc = LineCollection(paths, colors='blue', linewidths=0.5, alpha=0.4)
            ax.add_collection(lc)

        # Plot initial dots (only if grid isn't too dense)
        if args.grid_size <= 100:
            ax.scatter(X0, Y0, color='black', s=2, zorder=3, alpha=0.5)

        ax.set_title(f"Trajectory Evolution (t={t_snap:.2f})", fontsize=14)
        ax.set_xlim(gp['x_min'], gp['x_max'])
        ax.set_ylim(gp['y_min'], gp['y_max'])
        ax.set_aspect('equal')
        
        fpath = os.path.join(frames_dir, f"frame_{f_idx:03d}.png")
        plt.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fpath}")

if __name__ == "__main__":
    main()

