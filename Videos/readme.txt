# STOD Time-Evolution Videos

This directory contains high-resolution videos showcasing the temporal evolution of the **Strong Trajectorial Ontological Differentiation (STOD)** and **FinSTOD** measures across all dynamical systems studied in the manuscript.

## Alphabetical Ordering
The videos are prefixed with letters (**A** through **E**) to ensure they appear in the same order as the systems are presented in the paper:
- **A**: Linear Hyperbolic Saddle Point
- **B**: Simple Pendulum
- **C**: Lorenz '63 System (Poincaré Surface of Section)
- **D**: Forced Duffing Oscillator
- **E**: Time-Dependent Double Gyre

---

## Video Types

For each system and integration direction (forward/backward), we provide different visualization formats:

### 1. Comparison Videos (`*_Comparison.mp4`)
These are multi-panel videos designed for direct benchmarking. They typically show:
- **Standard Indicators**: FTLE, FLI, and LD (Lagrangian Descriptors).
- **STOD/FinSTOD**: Our proposed measure, allowing for a side-by-side comparison of how the different indicators resolve the system's skeleton over time.

### 2. Solo Videos (`*_SoloRaw.mp4` and `*_SoloLog.mp4`)
These videos focus exclusively on the STOD/FinSTOD measure to provide a cleaner view of its specific dynamics:
- **SoloRaw**: Shows the raw STOD values. This is often the most direct representation of the path-identity differentiation.
- **SoloLog**: Shows the log-transformed values ($\log(1+s)$). This transformation is particularly useful for complex or chaotic systems, as it enhances the contrast of subtle internal structures and hidden filaments that might be compressed in the raw scale.

---

## What to Observe

When viewing these videos, pay particular attention to:
- **Early Emergence**: How STOD/FinSTOD often resolves the "skeleton" of the flow (separatrices, manifolds, attractors) earlier in the integration time than standard indicators.
- **Internal Structures**: The emergence of internal spirals in the pendulum or secondary filaments in the double gyre that are uniquely identified by the path-based logic.
- **Temporal Stability**: How the identified structures stabilize and sharpen as the integration duration ($\tau$) increases.

For details on the specific parameters (grid resolution, integration time, physical constants) used for each video, please refer to the corresponding configuration files in the `configs/` directory of the pipeline.
