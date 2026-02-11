# Figure 6d: Hybrid-OSECIR-dABM scaling

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for plotting results and the Python benchmark script.

## Files

- **C++ Simulations**:
    - `scaling_dabm_ode_hybrid.cpp`: Main simulation file for scaling of dABM, OSECIR and temporal-hybrid dependent on the population size (Fig. 6d).
- **Plotting**:
    - `plot_hybrid_scaling.py`: Generates Fig. 6d from outputs of scaling_dabm_ode_hybrid.cpp.

The files described above can be found under `https://github.com/SciCompMod/memilio/tree/paper-compare-abm-ide-lct-ode/cpp/examples/paper`.

## How to Run

1.  **Build C++ Benchmarks**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Run C++ Simulations**:
    Execute the compiled C++ simulation binary:
    ```bash
    ./bin/scaling_dabm_ode_hybrid
    ```

4.  **Generate Plots**:
    Use the Python scripts for visualization by setting the simulation output directory and executing e.g.:
    ```bash
    python3 plot_hybrid_scaling.py
    ```
    Note, that this script uses the general configuration file 'plottings_settins.py'.
