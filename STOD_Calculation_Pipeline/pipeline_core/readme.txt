# Pipeline Core (`pipeline_core/`)

This directory contains the Python-based logic that orchestrates the STOD analysis workflow.

## Components
- **`generate_trajectories.py`**: Integrates the ODEs defined in the `systems/` folder and discretizes the resulting paths into a grid of cell indices.
- **`sod_logic.py`**: Python implementation of the ontological differentiation rules (used for prototyping and coordination).
- **`aggregate_results.py`**: Collects the raw comparison scores from the C++ backend, performs segmented normalization (T/UC/UU), and exports the final scalar fields as NumPy arrays.
- **`visualize_fields.py`**: Utility for generating preliminary heatmaps and vector field plots for verification.
- **`run_worker.py`**: Orchestrator for distributed execution across multiple CPU cores or HPC nodes.
