#!/usr/bin/env python3
"""
Field Visualization Script for STOD Pipeline

Generates vector field visualizations (standard and orthogonal) for each system.
- Non-autonomous systems (Double Gyre, Duffing): time evolution of fields
- Autonomous systems (Linear/Nonlinear Saddle, Lorenz): single snapshot

Outputs:
- PNG streamline plots for each time point
- NPY data files with field components (X, Y, U, V)

Usage:
    python visualize_fields.py [--config CONFIG_PATH] [--output OUTPUT_DIR] [--density DENSITY] [--resolution RES]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module

# =============================================================================
# FIELD COMPUTATION HELPERS
# =============================================================================

def get_velocity_field_grid(system_module, physics_params, grid_params, t, use_orthogonal=False):
    """
    Compute the velocity field on a grid at time t.
    
    Returns:
        X, Y: meshgrid coordinates
        U, V: velocity components
    """
    # Import the system module
    sys_module = import_module(system_module)
    
    # Merge initial_z0 from grid_params into physics_params if not already present
    # This is needed for Lorenz Poincaré section which uses z0 as a fixed parameter
    merged_physics = physics_params.copy()
    if 'initial_z0' in grid_params and 'initial_z0' not in merged_physics:
        merged_physics['initial_z0'] = grid_params['initial_z0']
    
    # Create grid
    x_vals = np.linspace(grid_params['x_min'], grid_params['x_max'], grid_params.get('viz_resolution', 100))
    y_vals = np.linspace(grid_params['y_min'], grid_params['y_max'], grid_params.get('viz_resolution', 100))
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute velocity at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # All systems now use 2D state (Lorenz is now a 2D Poincaré section)
            state = np.array([X[i, j], Y[i, j]])
            
            vel = sys_module.get_velocity_field(state, t, merged_physics)
            
            if use_orthogonal:
                # Rotate 90 degrees counterclockwise: (u, v) -> (-v, u)
                U[i, j] = -vel[1]
                V[i, j] = vel[0]
            else:
                U[i, j] = vel[0]
                V[i, j] = vel[1]
    
    return X, Y, U, V


def is_autonomous_system(system_name):
    """
    Determine if a system is autonomous (time-independent).
    """
    autonomous_systems = [
        'hyperbolic_linear', 
        'nonlinear_saddle', 
        'lorenz',
        'licn'
    ]
    return any(s in system_name.lower() for s in autonomous_systems)


def is_3d_system(system_name):
    """
    Determine if a system is truly 3D (or higher dimensional).
    
    Note: Lorenz is now implemented as a 2D Poincaré section (z fixed),
    so it's no longer considered a 3D system for orthogonal field purposes.
    Only LiCN remains as a true 4D system where orthogonal mode is problematic.
    """
    # Only LiCN is a true higher-dimensional system
    # Lorenz is now a 2D Poincaré section (z fixed)
    three_d_systems = ['licn']
    return any(s in system_name.lower() for s in three_d_systems)


# =============================================================================
# DATA SAVING FUNCTIONS
# =============================================================================

def save_field_data(output_dir, field_type, t, X, Y, U, V, time_index=None):
    """
    Save field data as NPY files.
    
    Args:
        output_dir: Directory to save files
        field_type: 'standard', 'orthogonal', or 'combined'
        t: Time value
        X, Y: Grid coordinates
        U, V: Velocity components
        time_index: Optional index for time series (used in filenames)
    """
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create filename based on time
    if time_index is not None:
        base_name = f'{field_type}_t{t:.4f}_idx{time_index:04d}'
    else:
        base_name = f'{field_type}_t{t:.4f}'
    
    # Save individual arrays
    np.save(os.path.join(data_dir, f'{base_name}_X.npy'), X)
    np.save(os.path.join(data_dir, f'{base_name}_Y.npy'), Y)
    np.save(os.path.join(data_dir, f'{base_name}_U.npy'), U)
    np.save(os.path.join(data_dir, f'{base_name}_V.npy'), V)
    
    # Also save a combined dictionary for convenience
    combined = {
        'X': X,
        'Y': Y,
        'U': U,
        'V': V,
        't': t,
        'field_type': field_type
    }
    np.save(os.path.join(data_dir, f'{base_name}_combined.npy'), combined, allow_pickle=True)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_streamlines(X, Y, U, V, title, output_path, density=1.5, color='black', 
                     x_lim=None, y_lim=None, equal_aspect=True):
    """
    Plot streamlines of a vector field.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize for color mapping (optional)
    speed = np.sqrt(U**2 + V**2)
    
    ax.streamplot(X[0, :], Y[:, 0], U, V, color=color, linewidth=0.8, 
                  density=density, arrowsize=1.0)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if equal_aspect:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_quiver(X, Y, U, V, title, output_path, scale=None, color='black', 
                x_lim=None, y_lim=None, equal_aspect=True, subsample=1):
    """
    Plot quiver (arrow) plot of a vector field.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Subsample for clearer visualization
    X_sub = X[::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample]
    U_sub = U[::subsample, ::subsample]
    V_sub = V[::subsample, ::subsample]
    
    ax.quiver(X_sub, Y_sub, U_sub, V_sub, color=color, scale=scale)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if equal_aspect:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_field_frames(config, output_dir, viz_resolution=100, stream_density=1.5):
    """
    Generate field visualization frames for a system.
    
    For autonomous systems: single frame
    For non-autonomous systems: frames at each snapshot time
    
    Outputs both PNG plots and NPY data files.
    """
    system_module = config['system_module']
    system_name = system_module.split('.')[-1]
    physics_params = config['physics_params']
    grid_params = config['grid_params'].copy()
    time_params = config['time_params']
    
    # Use visualization_params from config if available
    viz_params = config.get('visualization_params', {})
    if viz_params.get('enabled', True) is False:
        print(f"[SKIP] Visualization disabled for {system_name}")
        return 0
    
    # Override with config values if present
    viz_resolution = viz_params.get('grid_resolution', viz_resolution)
    stream_density = viz_params.get('stream_density', stream_density)
    
    grid_params['viz_resolution'] = viz_resolution
    
    # Create output directories
    base_dir = os.path.join(output_dir, system_name)
    std_dir = os.path.join(base_dir, 'standard_field')
    orth_dir = os.path.join(base_dir, 'orthogonal_field')
    combined_dir = os.path.join(base_dir, 'combined_field')
    
    # Create all directories including data subdirectories
    for d in [std_dir, orth_dir, combined_dir]:
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, 'data'), exist_ok=True)
    
    # Check if this is a 3D system
    is_3d = is_3d_system(system_name)
    if is_3d:
        print(f"[WARNING] {system_name} is a 3D system.")
        print(f"          The orthogonal field for 3D systems creates non-physical dynamics")
        print(f"          (only x,y components are rotated while z dynamics remain unchanged).")
        print(f"          Orthogonal metrics for this system should be interpreted with caution.")
    
    # Determine time points
    # Check config first, then fallback to heuristic detection
    is_auto_config = viz_params.get('is_autonomous', None)
    
    # Handle various input types for is_autonomous
    if isinstance(is_auto_config, str):
        is_auto = is_auto_config.lower() not in ('false', 'no', '0', 'off', 'n')
    elif is_auto_config is None:
        is_auto = is_autonomous_system(system_name)
    else:
        is_auto = bool(is_auto_config)
    
    print(f"[DEBUG] is_autonomous config value: {is_auto_config!r} -> interpreted as: {is_auto}")
    
    if is_auto:
        time_points = [0.0]  # Single snapshot for autonomous systems
        print(f"[INFO] {system_name} is autonomous - generating single snapshot at t=0")
    else:
        time_points = time_params.get('snapshot_times', [0.0])
        if not time_points:
            time_points = [0.0]
            print(f"[WARNING] No snapshot_times found in config, using t=0")
        print(f"[INFO] {system_name} is non-autonomous - generating {len(time_points)} frames")
        print(f"[DEBUG] Time points: {time_points}")
    
    # Grid limits
    x_lim = (grid_params['x_min'], grid_params['x_max'])
    y_lim = (grid_params['y_min'], grid_params['y_max'])
    equal_aspect = 'lorenz' not in system_name.lower()
    
    # Colors
    COLOR_STD = 'black'
    COLOR_ORTH = 'red'
    
    # Process each time point
    for i, t in enumerate(time_points):
        print(f"  Processing t = {t:.4f} ({i+1}/{len(time_points)})")
        
        # Compute standard field
        X, Y, U_std, V_std = get_velocity_field_grid(
            system_module, physics_params, grid_params, t, use_orthogonal=False
        )
        
        # Compute orthogonal field
        _, _, U_orth, V_orth = get_velocity_field_grid(
            system_module, physics_params, grid_params, t, use_orthogonal=True
        )
        
        # ----- STANDARD FIELD -----
        std_title = f'{system_name} Standard Field (t={t:.3f})'
        std_plot_path = os.path.join(std_dir, f'frame_{i:04d}.png')
        plot_streamlines(X, Y, U_std, V_std, std_title, std_plot_path, 
                        density=stream_density, color=COLOR_STD,
                        x_lim=x_lim, y_lim=y_lim, equal_aspect=equal_aspect)
        
        # Save standard field data
        save_field_data(std_dir, 'standard', t, X, Y, U_std, V_std, time_index=i)
        
        # ----- ORTHOGONAL FIELD -----
        orth_title = f'{system_name} Orthogonal Field (t={t:.3f})'
        orth_plot_path = os.path.join(orth_dir, f'frame_{i:04d}.png')
        plot_streamlines(X, Y, U_orth, V_orth, orth_title, orth_plot_path, 
                        density=stream_density, color=COLOR_ORTH,
                        x_lim=x_lim, y_lim=y_lim, equal_aspect=equal_aspect)
        
        # Save orthogonal field data
        save_field_data(orth_dir, 'orthogonal', t, X, Y, U_orth, V_orth, time_index=i)
        
        # ----- COMBINED FIELD -----
        combined_title = f'{system_name} Combined Fields (t={t:.3f})'
        combined_plot_path = os.path.join(combined_dir, f'frame_{i:04d}.png')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        # Note: streamplot doesn't support alpha directly, use lighter colors instead
        ax.streamplot(X[0, :], Y[:, 0], U_std, V_std, color=(0.3, 0.3, 0.3), 
                     linewidth=0.6, density=stream_density * 0.7)
        ax.streamplot(X[0, :], Y[:, 0], U_orth, V_orth, color=(0.8, 0.2, 0.2), 
                     linewidth=0.6, density=stream_density * 0.7)
        ax.set_title(combined_title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        if equal_aspect:
            ax.set_aspect('equal')
        ax.legend(['Standard', 'Orthogonal'], loc='upper right')
        plt.tight_layout()
        plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary/metadata file
    metadata = {
        'system_name': system_name,
        'system_module': system_module,
        'physics_params': physics_params,
        'grid_params': {k: v for k, v in grid_params.items() if not isinstance(v, np.ndarray)},
        'time_points': time_points,
        'is_autonomous': is_auto,
        'is_3d_system': is_3d,
        'viz_resolution': viz_resolution,
        'stream_density': stream_density,
        'n_frames': len(time_points)
    }
    
    # Save metadata
    metadata_path = os.path.join(base_dir, 'metadata.npy')
    np.save(metadata_path, metadata, allow_pickle=True)
    
    # Also save as YAML for human readability
    metadata_yaml_path = os.path.join(base_dir, 'metadata.yaml')
    # Convert numpy types to native Python for YAML
    metadata_yaml = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                    for k, v in metadata.items()}
    metadata_yaml['time_points'] = [float(t) for t in time_points]
    with open(metadata_yaml_path, 'w') as f:
        yaml.dump(metadata_yaml, f, default_flow_style=False)
    
    print(f"[DONE] Generated {len(time_points)} frame(s) for {system_name}")
    print(f"       Plots saved in: {base_dir}")
    print(f"       Data files saved in: {std_dir}/data, {orth_dir}/data")
    
    return len(time_points)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate vector field visualizations for STOD pipeline'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to system configuration YAML file')
    parser.add_argument('--output', type=str, default='field_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--density', type=float, default=1.5,
                       help='Streamline density (default: 1.5)')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Grid resolution for visualization (default: 100)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        # Try looking in configs folder
        alt_path = os.path.join('configs', config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("STOD Pipeline Field Visualization")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {args.output}")
    print(f"Density: {args.density}")
    print(f"Resolution: {args.resolution}")
    print("=" * 60)
    
    # Generate visualizations
    n_frames = generate_field_frames(
        config, 
        args.output,
        viz_resolution=args.resolution,
        stream_density=args.density
    )
    
    print("=" * 60)
    print(f"Visualization complete! {n_frames} frame(s) generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()

