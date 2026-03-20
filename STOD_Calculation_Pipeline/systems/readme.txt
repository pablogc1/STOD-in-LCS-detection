# Dynamical Systems (`systems/`)

This directory contains the mathematical definitions of the dynamical systems studied in the paper.

## Implementation
Each system is implemented as a Python module that provides the velocity field (vector field) for the ODE integrator.

- **`linear_saddle.py`**: 2D linear hyperbolic saddle point.
- **`pendulum.py`**: Simple pendulum (nonlinear autonomous).
- **`lorenz.py`**: Lorenz '63 system (3D chaotic, analyzed via Poincaré section).
- **`duffing.py`**: Forced Duffing oscillator (non-autonomous chaotic).
- **`doublegyre.py`**: Time-dependent double gyre (non-autonomous mixing flow).

## Adding a New System
To analyze a new system:
1. Create a new `.py` file in this directory defining the equations of motion.
2. Register the system in the `pipeline_core` integrator.
3. Create a corresponding configuration file in `configs/`.
