# C++ Backend (`cpp_backend/`)

The core computational engine of the STOD pipeline, implemented in C++ for maximum performance during trajectory comparison.

## Overview
Comparing millions of trajectory histories (each consisting of thousands of cell indices) is a computationally intensive task. This backend performs the **Trajectorial Ontological Differentiation (TOD)** logic at high speed.

- **`calc_metrics.cpp`**: The main engine that loads discretized trajectory data and performs neighbor-to-neighbor path comparisons.
- **`sod_logic.hpp`**: Header file containing the core comparison algorithms and cancellation rules.
- **`sod_canary.cpp`**: Validation tool used to verify the consistency of the comparison logic.

## Compilation
The backend must be compiled before running the pipeline:
```bash
g++ -O3 calc_metrics.cpp -o calc_metrics
```
The `-O3` flag is highly recommended for production runs on HPC clusters like Magerit.
