# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: Lorenz '63 (Full 3D System)
# ----------------------------------------------------------------------
# This file provides the "scientific personality" for the Lorenz system.
# 
# This implements the FULL 3D Lorenz '63 chaotic attractor.
# z evolves dynamically according to dz/dt = xy - βz.
# The 2D analysis grid is (x, y), with trajectories starting from z = z0.
#
# Equations:
#   dx/dt = σ(y - x)
#   dy/dt = x(ρ - z) - y
#   dz/dt = xy - βz
#
# It adheres to the contract required by the generic pipeline:
#   - get_velocity_field(...)
#   - get_initial_state_from_grid(...)
#   - discretize_trajectory_to_grid(...)
# ----------------------------------------------------------------------

import numpy as np

# ======================================================================
#         1. VELOCITY FIELD (THE CORE PHYSICS)
# ======================================================================

def get_velocity_field(state, t, physics_params):
    """
    The Lorenz '63 ordinary differential equations (full 3D system).
    
    Args:
        state (np.ndarray): The current state vector [x, y, z].
        t (float): The current time (unused for Lorenz '63, but required by integrators).
        physics_params (dict): A dictionary containing the system's parameters,
                               loaded directly from the YAML config file.
                               Must include 'sigma', 'rho', 'beta'.
    
    Returns:
        np.ndarray: The velocity vector [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state[0], state[1], state[2]
    
    # Unpack parameters from the config dictionary
    sigma = physics_params['sigma']
    rho = physics_params['rho']
    beta = physics_params['beta']
    
    # Full Lorenz equations
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return np.array([dx_dt, dy_dt, dz_dt])


# ======================================================================
#         2. GRID-TO-STATE MAPPING
# ======================================================================

def get_initial_state_from_grid(row, col, grid_params):
    """
    Maps a 2D grid cell index (row, col) to a 3D physical initial state.
    For this system, we create a 2D grid in (x, y) on a fixed z=z0 plane.
    
    Args:
        row (int): The row index of the grid cell.
        col (int): The column index of the grid cell.
        grid_params (dict): Dictionary of grid parameters from the YAML config.
        
    Returns:
        np.ndarray: The 3D initial state vector [x0, y0, z0].
    """
    # Unpack grid parameters from the config dictionary
    nx = grid_params['grid_resolution_x']
    ny = grid_params['grid_resolution_y']
    x_min, x_max = grid_params['x_min'], grid_params['x_max']
    y_min, y_max = grid_params['y_min'], grid_params['y_max']
    z0 = grid_params['initial_z0']  # The fixed z-plane for starting trajectories

    # Calculate the center of the grid cell
    x = x_min + (col + 0.5) * (x_max - x_min) / nx
    y = y_min + (row + 0.5) * (y_max - y_min) / ny
    
    return np.array([x, y, z0])


# ======================================================================
#         3. TRAJECTORY-TO-GRID MAPPING
# ======================================================================

def discretize_trajectory_to_grid(trajectory, grid_params):
    """
    Projects a 3D trajectory onto the 2D analysis grid.
    For this system, we take the (x, y) components of the 3D path
    and determine which grid cells they fall into.
    
    Args:
        trajectory (np.ndarray): A (num_steps, 3) array of the path [x, y, z].
        grid_params (dict): Dictionary of grid parameters from the YAML config.
        
    Returns:
        list[tuple[int, int]]: A list of (row, col) tuples representing the
                               discretized path on the grid.
    """
    # Unpack grid parameters
    nx = grid_params['grid_resolution_x']
    ny = grid_params['grid_resolution_y']
    x_min, x_max = grid_params['x_min'], grid_params['x_max']
    y_min, y_max = grid_params['y_min'], grid_params['y_max']

    # Extract only the x and y components (ignore z for grid projection)
    x_coords, y_coords = trajectory[:, 0], trajectory[:, 1]
    
    # Convert physical coordinates to grid indices
    col_indices = ((x_coords - x_min) / (x_max - x_min) * nx).astype(int)
    row_indices = ((y_coords - y_min) / (y_max - y_min) * ny).astype(int)

    # Ensure indices stay within the grid bounds [0, N-1]
    col_indices = np.clip(col_indices, 0, nx - 1)
    row_indices = np.clip(row_indices, 0, ny - 1)
    
    return list(zip(row_indices, col_indices))
