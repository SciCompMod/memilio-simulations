# Figure 6a: ODE SEIR Metapopulation Benchmark

This folder contains the benchmarks comparing the C++ implementation of the SEIR metapopulation model against R implementations.

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **R**: Required for running the R benchmarks.
- **Python**: Required for plotting results and the Python benchmark script.

## Files

- **C++ Simulations**:
    - `ode_seir_metapop.cpp`: Main simulation file.
    - `ode_seir_metapop_benchmark.cpp`: Benchmark implementation.
- **R Simulations**:
    - `ode_seir_metapop.R`: R implementation of the model.
    - `benchmark_seir_metapop_R*.R`: Benchmark scripts for R (Euler, RK4).
- **Python Simulations**:
    - `ode_seir_metapop_benchmark.py`: Benchmark implementation using the Python bindings.
- **Plotting**:
    - `plot_speedup_seir_metapop.py`: Generic script to plot speedup results.

The implementation of the model can be found under `https://github.com/SciCompMod/memilio/tree/1003-ode-seir-metapop-add-bindings`.

## How to Run

1.  **Build C++ Benchmarks**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Run C++ Benchmark**:
    Execute the compiled C++ benchmark binary:
    ```bash
    ./bin/ode_seir_metapop_benchmark
    ```

3.  **Run R Benchmarks**:
    Execute the R scripts to generate the R benchmark data:
    ```bash
    Rscript benchmark_seir_metapop_R_explicit_euler.R
    Rscript benchmark_seir_metapop_R_rk4.R
    ```

3.  **Run Python Benchmarks**:
    Follow the installation steps for the MEmilio Python bindings as described in the Memilio documentation.
    Execute the Python benchmark script:
    ```bash
    python3 ode_seir_metapop_benchmark.py
    ```

4.  **Generate Plots**:
    Use the Python script to visualize the comparison:
    ```bash
    python3 plot_speedup_seir_metapop.py
    ```
    Note, that this script uses the general configuration file 'plottings_settins.py' in the parent folder.

This will generate the runtime and speedup plots (e.g., `runtime_seir_metapop.png`, `speedup_total_seir_metapop.png`).
