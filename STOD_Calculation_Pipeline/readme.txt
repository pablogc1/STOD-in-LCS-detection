# STOD Calculation Pipeline

This folder contains the core computational pipeline for calculating **Strong Trajectorial Ontological Differentiation (STOD)** and its variants (like FinSTOD).

## High-Performance Computing (HPC) Context

The pipeline is designed for high-throughput trajectory analysis and is optimized for execution on **High-Performance Computing (HPC) clusters**, specifically the **Magerit Supercomputer** at **CeSViMa** (Centro de Supercomputación y Visualización de Madrid).

Due to the exhaustive nature of path-identity comparison, which requires storing and comparing complete trajectory histories for millions of initial conditions, we leverage a distributed computing approach using **SLURM** for job scheduling and a high-performance **C++ backend** for the core metric calculations.

---

## Pipeline Architecture

The pipeline follows a modular design:
1.  **Configuration (`configs/`)**: All system parameters, grid resolutions, and integration times are defined in YAML files.
2.  **Trajectory Generation (`pipeline_core/`)**: Python scripts handle the ODE integration and path discretization.
3.  **Core Calculation (`cpp_backend/`)**: A dedicated C++ engine performs the iterative ontological differentiation between trajectory pairs.
4.  **Aggregation (`pipeline_core/`)**: Results are collected, normalized, and stored as NumPy arrays for visualization.

---

## How to Run the Pipeline

### 1. Configure the System
All parameters are controlled via a single YAML configuration file. For example, to modify the Pendulum analysis, edit `configs/pendulum_config.yaml`:

```yaml
system_name: "pendulum"
grid_resolution_x: 1000
grid_resolution_y: 1000
T_max: 20.0
# ... other parameters ...
```

### 2. Local Execution (Testing)
For small-scale tests or debugging, you can run the master script directly:

```bash
./master_run.sh configs/your_config.yaml
```

### 3. HPC Execution (CeSViMa / SLURM)
For full-resolution production runs (e.g., $1000 \times 1000$ or $2000 \times 1000$ grids), use the provided SLURM submission script:

```bash
sbatch sbatch_trajectory_export.slurm configs/your_config.yaml
```

This script is pre-configured for the Magerit environment, handling node allocation, environment setup, and parallel execution across the cluster.

---

## Workflow Summary
1.  **Integrate**: Trajectories are integrated and discretized into a grid of cell indices.
2.  **Compare**: The C++ backend compares each trajectory against its neighbors to calculate the STOD/FinSTOD score.
3.  **Export**: The resulting scalar fields are saved as `.npy` files, which are then used by the `Latest_Figure_Generator.py` script to produce the paper figures.

For detailed descriptions of the individual metrics and comparison rules, please refer to `METRIC_DESCRIPTIONS.md` in the root directory.
