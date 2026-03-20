#!/usr/bin/env python3
"""
Merge Trajectories - Combines partial HDF5 trajectory files into one
====================================================================
This script merges all chunk_XXXX.h5 files from the trajectory_cache/partial
folder into a single trajectories_merged.h5 file for easy download.

Usage:
    python3 merge_trajectories.py <output_base_folder>

Example:
    python3 merge_trajectories.py trajectory_export_output

Output:
    <output_base_folder>/trajectories_merged.h5

The merged file contains:
    - forward: shape (grid_y, grid_x, n_steps, n_dims)
    - backward: shape (grid_y, grid_x, n_steps, n_dims)
    - metadata: grid parameters, time parameters, etc.
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
import yaml
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Merge partial trajectory HDF5 files into one"
    )
    parser.add_argument("output_base_folder", type=str,
                        help="Path to the output base folder (e.g., trajectory_export_output)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (auto-detected if not provided)")
    args = parser.parse_args()

    output_base = args.output_base_folder
    partial_folder = os.path.join(output_base, "trajectory_cache", "partial")
    output_file = os.path.join(output_base, "trajectories_merged.h5")

    print("=" * 70)
    print("  TRAJECTORY MERGE UTILITY")
    print("=" * 70)

    # Find all partial files
    partial_files = sorted(glob.glob(os.path.join(partial_folder, "chunk_*.h5")))
    if not partial_files:
        print(f"ERROR: No chunk_*.h5 files found in {partial_folder}")
        sys.exit(1)

    print(f"\nFound {len(partial_files)} partial files in {partial_folder}")

    # Try to find config file
    config_file = args.config
    if not config_file:
        # Try common locations
        possible_configs = [
            os.environ.get('SYSTEM_CONFIG_FILE', ''),
            os.path.join(os.path.dirname(output_base), 'configs', 'trajectory_export_template.yaml'),
        ]
        for cfg in possible_configs:
            if cfg and os.path.exists(cfg):
                config_file = cfg
                break

    config = None
    if config_file and os.path.exists(config_file):
        print(f"Using config: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

    # Read first file to get dimensions
    print("\nAnalyzing file structure...")
    with h5py.File(partial_files[0], 'r') as f:
        first_shape = f['forward'].shape  # (rows_in_chunk, nx, steps, dims)
        n_dims = first_shape[3]
        n_steps = first_shape[2]
        nx = first_shape[1]

    # Calculate total rows
    total_rows = 0
    chunk_info = []
    for fpath in partial_files:
        with h5py.File(fpath, 'r') as f:
            chunk_rows = f['forward'].shape[0]
            chunk_info.append((fpath, total_rows, chunk_rows))
            total_rows += chunk_rows

    ny = total_rows
    print(f"  Grid: {nx} × {ny}")
    print(f"  Time steps: {n_steps}")
    print(f"  Dimensions: {n_dims}")

    # Extract grid and time info from config if available
    if config:
        gp = config.get('grid_params', {})
        tp = config.get('time_params', {})
        x_min = gp.get('x_min', -1.0)
        x_max = gp.get('x_max', 1.0)
        y_min = gp.get('y_min', -1.0)
        y_max = gp.get('y_max', 1.0)
        t_end = tp.get('total_integration_time_end', 10.0)
        dt = tp.get('time_step_dt', 0.1)
        system_module = config.get('system_module', 'unknown')
        field_mode = config.get('field_mode', 'standard')
        physics_params = config.get('physics_params', {})
    else:
        # Use defaults
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        t_end = 10.0
        dt = 0.1
        system_module = 'unknown'
        field_mode = 'standard'
        physics_params = {}

    # Create merged file
    print(f"\nCreating merged file: {output_file}")
    with h5py.File(output_file, 'w') as fout:
        # Create datasets
        forward_ds = fout.create_dataset(
            'forward',
            shape=(ny, nx, n_steps, n_dims),
            dtype=np.float64,
            compression='gzip',
            compression_opts=4,
            chunks=(1, nx, n_steps, n_dims)
        )
        backward_ds = fout.create_dataset(
            'backward',
            shape=(ny, nx, n_steps, n_dims),
            dtype=np.float64,
            compression='gzip',
            compression_opts=4,
            chunks=(1, nx, n_steps, n_dims)
        )

        # Copy data from each chunk
        for i, (fpath, start_row, chunk_rows) in enumerate(chunk_info):
            print(f"  Processing {os.path.basename(fpath)} ({i+1}/{len(chunk_info)})...")
            with h5py.File(fpath, 'r') as fin:
                forward_ds[start_row:start_row+chunk_rows, :, :, :] = fin['forward'][:]
                backward_ds[start_row:start_row+chunk_rows, :, :, :] = fin['backward'][:]

        # Add metadata
        meta = fout.create_group('metadata')
        meta.attrs['grid_resolution_x'] = nx
        meta.attrs['grid_resolution_y'] = ny
        meta.attrs['x_min'] = x_min
        meta.attrs['x_max'] = x_max
        meta.attrs['y_min'] = y_min
        meta.attrs['y_max'] = y_max
        meta.attrs['t_end'] = t_end
        meta.attrs['dt'] = dt
        meta.attrs['n_steps'] = n_steps
        meta.attrs['n_dims'] = n_dims
        meta.attrs['system_module'] = system_module
        meta.attrs['field_mode'] = field_mode
        meta.attrs['created'] = datetime.now().isoformat()

        # Add physics params as a sub-group
        phys_grp = meta.create_group('physics_params')
        for key, val in physics_params.items():
            if isinstance(val, (int, float)):
                phys_grp.attrs[key] = val

        # Create coordinate arrays for convenience
        # Grid cell centers (matching C++ convention: (i + 0.5) * delta)
        x_centers = np.linspace(x_min, x_max, nx, endpoint=False) + (x_max - x_min) / (2 * nx)
        y_centers = np.linspace(y_min, y_max, ny, endpoint=False) + (y_max - y_min) / (2 * ny)
        t_array = np.linspace(0, t_end, n_steps)

        fout.create_dataset('x_coords', data=x_centers, compression='gzip')
        fout.create_dataset('y_coords', data=y_centers, compression='gzip')
        fout.create_dataset('t_array', data=t_array, compression='gzip')

        # Create 2D initial condition grids
        X0, Y0 = np.meshgrid(x_centers, y_centers)
        fout.create_dataset('x0', data=X0, compression='gzip')
        fout.create_dataset('y0', data=Y0, compression='gzip')

        # Find indices of points on axes
        on_x_axis = np.where(np.abs(Y0.ravel()) < (y_max - y_min) / (2 * ny))[0]
        on_y_axis = np.where(np.abs(X0.ravel()) < (x_max - x_min) / (2 * nx))[0]
        fout.create_dataset('indices_on_x_axis', data=on_x_axis)
        fout.create_dataset('indices_on_y_axis', data=on_y_axis)

    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("  MERGE COMPLETE")
    print("=" * 70)
    print(f"\n  Output file: {output_file}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"\n  Data structure:")
    print(f"    forward:  ({ny}, {nx}, {n_steps}, {n_dims})")
    print(f"    backward: ({ny}, {nx}, {n_steps}, {n_dims})")
    print(f"    x_coords: ({nx},)")
    print(f"    y_coords: ({ny},)")
    print(f"    t_array:  ({n_steps},)")
    print(f"    x0, y0:   ({ny}, {nx})")
    print("\n  Download this file and use plot_trajectories_spyder.py to visualize!")
    print("=" * 70)


if __name__ == "__main__":
    main()

