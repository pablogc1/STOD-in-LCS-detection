# Configuration (`configs/`)

This directory contains the YAML configuration files that define the parameters for each dynamical system analyzed in the STOD framework.

## Usage
Each file controls the physical parameters of the system, the grid resolution for the phase space discretization, and the integration settings.

- **`system_name`**: Identifier for the system (e.g., `pendulum`, `duffing`).
- **`grid_resolution_x`, `grid_resolution_y`**: Number of initial conditions along each axis.
- **`x_min`, `x_max`, `y_min`, `y_max`**: Domain boundaries in phase space.
- **`T_max`**: Total integration time for the trajectories.
- **`dt`**: Time step for the ODE solver.
- **`system_params`**: Physical constants specific to the model (e.g., `alpha`, `sigma`, `rho`).

To run a new experiment, simply create or modify a `.yaml` file and pass it as an argument to the `master_run.sh` script.
