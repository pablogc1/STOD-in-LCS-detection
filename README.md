# Strong Trajectorial Ontological Differentiation (STOD) - Official Repository

This repository contains the source code, configuration files, and complete visual results for the framework introduced in the paper: **"Strong Trajectorial Ontological Differentiation: A novel approach to unravel phase-space structures"**.

## Overview

**Strong Trajectorial Ontological Differentiation (STOD)** is a novel trajectory-based framework that reframes dynamical divergence through the lens of ontological similarity. Unlike traditional variational indicators (such as FTLE or FLI) that rely on local linearizations (Jacobians) or integral-based methods (like LD), STOD treats the complete path of a particle as its unique identity.

By performing a component-wise cancellation process between these path identities, STOD quantifies the similarity between trajectories, providing a robust and system-agnostic measure of the structures that govern transport and mixing in complex flows.

### Key Features
- **System-Agnostic**: Does not require Jacobians or specific velocity integrals.
- **High Resolution**: Resolves fine-scale filamentary structures and subtle path connections.
- **Direction Sensitive**: Naturally distinguishes between forward and backward temporal orientations.
- **Robust**: Effective in both simple linear regimes and high-dimensional chaotic flows.

---

## Repository Structure

### 1. The STOD Pipeline
The core implementation of the framework, designed for high-performance computation and scalability.
- **`pipeline_core/`**: Python-based logic for trajectory generation, path comparison, and result aggregation.
- **`cpp_backend/`**: C++ implementation of the core comparison algorithms for maximum efficiency.
- **`configs/`**: YAML configuration files for all systems studied in the paper (Linear Saddle, Pendulum, Lorenz '63, Duffing, and Double Gyre).
- **`master_run.sh`**: Main entry point to execute the full analysis pipeline.
  
The Python scripts used to generate the publication-ready figures found in the manuscript:
- **`Latest_Figure_Generator.py`**: Generates the main comparative grids and 2x2 plots.
- **`Snake_FinSTOD.py`**: Generates the worked example of the serpentine path.
  
### 2. Full Time-Evolution Videos (`/Videos`)
While the paper presents static snapshots, the true power of STOD is best observed through its temporal evolution. This folder contains high-resolution videos comparing STOD/FinSTOD against standard indicators (FTLE, FLI, LD) across the entire integration interval for all systems.

The videos are organized as follows:
- **A_Linear_Saddle_...**: Evolution of the linear hyperbolic saddle point.
- **B_Simple_Pendulum_...**: Evolution of the pendulum's phase space, showing the emergence of internal spirals.
- **C_Lorenz_63_...**: Evolution on the $z=27.0$ Poincaré section, revealing the strange attractor's skeleton.
- **D_Forced_Duffing_...**: Evolution of the chaotic attractor and its fractal boundaries.
- **E_Double_Gyre_...**: Evolution of chaotic mixing in a non-autonomous flow.


For any questions regarding the implementation or the STOD framework, please refer to the contact information provided in the manuscript.

## Acknowledgments and AI Assistance

The author is fully responsible for the conceptual framework, the implementation of the STOD pipeline, and the final results presented in the manuscript. 

However, the development of this repository and the refinement of the manuscript were supported by advanced AI agents, specifically **Gemini 3 Flash** and **Claude 4.5 Opus**, which assisted in code optimization, documentation, and technical writing. All AI-assisted contributions were critically reviewed and verified by the author.
