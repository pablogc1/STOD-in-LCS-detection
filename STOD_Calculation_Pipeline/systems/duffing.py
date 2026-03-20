# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: Forced Duffing Oscillator (2D: x, v)
# ----------------------------------------------------------------------
# Contract:
#   - get_velocity_field(state, t, physics_params) -> np.ndarray([dx/dt, dv/dt])
#   - get_initial_state_from_grid(row, col, grid_params) -> np.ndarray([x0, v0])
#   - discretize_trajectory_to_grid(trajectory, grid_params) -> list[(r,c)]
# ----------------------------------------------------------------------

import numpy as np

def get_velocity_field(state, t, physics_params):
    """
    Duffing ODE:
        x' = v
        v' = gamma*cos(omega*t) - delta*v - alpha*x - beta*x^3
    physics_params must include: alpha, beta, delta, gamma, omega
    """
    x, v = float(state[0]), float(state[1])
    alpha = float(physics_params['alpha'])
    beta  = float(physics_params['beta'])
    delta = float(physics_params['delta'])
    gamma = float(physics_params['gamma'])
    omega = float(physics_params['omega'])

    dxdt = v
    dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * (x ** 3)
    return np.array([dxdt, dvdt], dtype=float)

def get_initial_state_from_grid(row, col, grid_params):
    """
    Map grid cell center (row, col) -> initial state [x0, v0].
    Uses the generic grid names (x,y) where y ≡ v here.
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    v_min, v_max = float(grid_params['y_min']), float(grid_params['y_max'])  # y-axis is velocity

    x0 = x_min + (col + 0.5) * (x_max - x_min) / nx
    v0 = v_min + (row + 0.5) * (v_max - v_min) / ny
    return np.array([x0, v0], dtype=float)

def discretize_trajectory_to_grid(trajectory, grid_params):
    """
    Project continuous (x,v) trajectory onto the analysis grid as (row, col) cells.
    Keeps duplicates (one cell per time sample) to preserve SOD semantics.
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    v_min, v_max = float(grid_params['y_min']), float(grid_params['y_max'])

    x = trajectory[:, 0]
    v = trajectory[:, 1]

    cols = ((x - x_min) / (x_max - x_min) * nx).astype(int)
    rows = ((v - v_min) / (v_max - v_min) * ny).astype(int)

    cols = np.clip(cols, 0, nx - 1)
    rows = np.clip(rows, 0, ny - 1)

    return list(zip(rows, cols))

