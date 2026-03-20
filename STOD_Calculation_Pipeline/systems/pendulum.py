# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: Simple Pendulum (2D Phase Space)
# ----------------------------------------------------------------------
# Contract required by the generic pipeline:
#   - get_velocity_field(state, t, physics_params) -> np.ndarray([u, v])
#   - get_initial_state_from_grid(row, col, grid_params) -> np.ndarray([x0, y0])
#   - discretize_trajectory_to_grid(trajectory, grid_params) -> list[(r,c)]
#
# State is 2D: [theta, omega] where:
#   theta = angular position
#   omega = angular velocity (dtheta/dt)
#
# Equations of motion:
#   dtheta/dt = omega
#   domega/dt = -lambda * sin(theta)
#
# where lambda = g/L (gravitational acceleration / pendulum length)
#
# Key features:
#   - Stable equilibrium at (0, 0) - elliptic fixed point
#   - Unstable equilibrium at (±π, 0) - hyperbolic saddle
#   - Separatrix at energy E = lambda (in non-dimensional units)
# ----------------------------------------------------------------------

import numpy as np


def get_velocity_field(state, t, physics_params):
    """
    Simple pendulum phase space dynamics:
        dtheta/dt = omega
        domega/dt = -lambda * sin(theta)

    Parameters
    ----------
    state : array_like, shape (2,)
        [theta, omega] - angular position and velocity
    t : float
        time (unused - autonomous system)
    physics_params : dict
        {'lambda': float} - ratio g/L

    Returns
    -------
    np.ndarray, shape (2,)
        [dtheta/dt, domega/dt]
    """
    theta, omega = float(state[0]), float(state[1])
    lam = float(physics_params['lambda'])

    dtheta = omega
    domega = -lam * np.sin(theta)
    return np.array([dtheta, domega], dtype=float)


def get_initial_state_from_grid(row, col, grid_params):
    """
    Map grid cell center (row, col) -> initial state [theta0, omega0].
    
    Grid convention:
        x-axis (cols) = theta (angular position)
        y-axis (rows) = omega (angular velocity)
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    y_min, y_max = float(grid_params['y_min']), float(grid_params['y_max'])

    theta0 = x_min + (col + 0.5) * (x_max - x_min) / nx
    omega0 = y_min + (row + 0.5) * (y_max - y_min) / ny
    return np.array([theta0, omega0], dtype=float)


def discretize_trajectory_to_grid(trajectory, grid_params):
    """
    Project continuous (theta, omega) trajectory onto the analysis grid as (row, col) cells.
    Keeps duplicates (one cell per time sample) to preserve SOD semantics.
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    x_min, x_max = float(grid_params['x_min']), float(grid_params['x_max'])
    y_min, y_max = float(grid_params['y_min']), float(grid_params['y_max'])

    theta = trajectory[:, 0]
    omega = trajectory[:, 1]

    cols = ((theta - x_min) / (x_max - x_min) * nx).astype(int)
    rows = ((omega - y_min) / (y_max - y_min) * ny).astype(int)

    cols = np.clip(cols, 0, nx - 1)
    rows = np.clip(rows, 0, ny - 1)

    return list(zip(rows, cols))
