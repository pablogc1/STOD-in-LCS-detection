# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: Linear Hyperbolic Saddle (2D)
# ----------------------------------------------------------------------
# Contract required by the generic pipeline:
#   - get_velocity_field(state, t, physics_params) -> np.ndarray([u, v])
#   - get_initial_state_from_grid(row, col, grid_params) -> np.ndarray([x0, y0])
#   - discretize_trajectory_to_grid(trajectory, grid_params) -> list[(r,c)]
# State is 2D: [x, y]
# ----------------------------------------------------------------------

import numpy as np

def get_velocity_field(state, t, physics_params):
    """
    Linear hyperbolic saddle:
        dx/dt = +alpha * x
        dy/dt = -alpha * y
    """
    x, y = float(state[0]), float(state[1])
    alpha = float(physics_params["alpha"])
    u =  alpha * x
    v = -alpha * y
    return np.array([u, v], dtype=float)

def get_initial_state_from_grid(row, col, grid_params):
    """
    Map grid cell center (row, col) -> 2D initial state [x0, y0].
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    y_min, y_max = float(grid_params['y_min']), float(grid_params['y_max'])

    x = x_min + (col + 0.5) * (x_max - x_min) / nx
    y = y_min + (row + 0.5) * (y_max - y_min) / ny
    return np.array([x, y], dtype=float)

def discretize_trajectory_to_grid(trajectory, grid_params):
    """
    Project continuous 2D trajectory onto the analysis grid as (row, col) cells.
    Keeps duplicates (one cell per time sample) to match SOD semantics.
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    y_min, y_max = float(grid_params['y_min']), float(grid_params['y_max'])

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    cols = ((x - x_min) / (x_max - x_min) * nx).astype(int)
    rows = ((y - y_min) / (y_max - y_min) * ny).astype(int)

    cols = np.clip(cols, 0, nx - 1)
    rows = np.clip(rows, 0, ny - 1)

    return list(zip(rows, cols))

