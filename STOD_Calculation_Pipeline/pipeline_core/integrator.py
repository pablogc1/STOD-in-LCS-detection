# ----------------------------------------------------------------------
#        PIPELINE CORE: Generic ODE Integrator (odeint)
# ----------------------------------------------------------------------

import numpy as np
from scipy.integrate import odeint

def generate_trajectory(velocity_func, y0, t_end, dt, direction, physics_params):
    """
    Integrates a trajectory from an initial condition y0 using odeint.

    velocity_func(state, t, physics_params) -> dstate/dt
    direction: +1 forward (0->t_end), -1 backward (t_end->0)
    """
    y0 = np.asarray(y0, dtype=float)

    if direction > 0:
        t_eval = np.arange(0.0, t_end + dt, dt, dtype=float)
    else:
        t_eval = np.arange(t_end, -dt, -dt, dtype=float)

    # odeint expects f(y, t)
    def f(y, t):
        return velocity_func(y, t, physics_params)

    traj = odeint(f, y0, t_eval)
    return traj  # shape (num_steps, num_dims)

