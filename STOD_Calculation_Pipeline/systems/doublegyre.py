# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: Time-Dependent Double Gyre (2D)
# ----------------------------------------------------------------------
# Contract required by the generic pipeline:
#   - get_velocity_field(state, t, physics_params) -> np.ndarray([u, v])
#   - get_initial_state_from_grid(row, col, grid_params) -> np.ndarray([x0, y0])
#   - discretize_trajectory_to_grid(trajectory, grid_params) -> list[(r,c)]
# State is 2D: [x, y]
# ----------------------------------------------------------------------

import numpy as np

def _a(t, epsilon, omega):
    return epsilon * np.sin(omega * t)

def _b(t, epsilon, omega):
    return 1.0 - 2.0 * _a(t, epsilon, omega)

def _f(x, t, epsilon, omega):
    return _a(t, epsilon, omega) * x * x + _b(t, epsilon, omega) * x

def get_velocity_field(state, t, physics_params):
    """
    Double Gyre velocity field at (x, y, t).

    Args:
        state: np.ndarray([x, y])  (2D state)
        t: float
        physics_params: dict with keys {'A','epsilon','omega'}

    Returns:
        np.ndarray([u, v])
    """
    x, y = float(state[0]), float(state[1])
    A = float(physics_params['A'])
    eps = float(physics_params['epsilon'])
    w = float(physics_params['omega'])

    f = _f(x, t, eps, w)
    dfdx = 2.0 * _a(t, eps, w) * x + _b(t, eps, w)

    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx
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

