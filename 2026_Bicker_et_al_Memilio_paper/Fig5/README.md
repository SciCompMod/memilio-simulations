# Figure 5: Hybrid-OSECIR-dABM comparison and ABM-IDE-LCT-ODE comparison

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for plotting results and the Python benchmark script.

## Files

- **C++ Simulations**:
    - `comparison_dabm_ode_hybrid.cpp`: Main simulation file for scenario comparison of dABM, OSECIR and temporal-hybrid (Fig. 5b and c).
    - `abm_ide_lct_ode.cpp`: Main simulation file for scenario comparison of ABM, ODE-SECIR, LCT-SECIR and IDE-SECIR (Fig. 5d and e).
- **Python Simulations**:
    - `get_lognormal_and_erlang_parameters.py`: Returns for given mean and variance the corresponding Lognormal and Erlang parameters used for S2 in Fig. 5e.
- **Plotting**:
    - `plot_hybrid_application.py`: Generates Fig. 5b+c from outputs of comparison_dabm_ode_hybrid.cpp.
    - `plot_model_comparison.py`: Generates Fig. 5e from outputs of abm_ide_lct_ode.cpp.
    - `plot_pdfs.py`: Plots probability density function of distributions used for Fig.5e.

The files described above can be found under `https://github.com/SciCompMod/memilio/tree/paper-compare-abm-ide-lct-ode/cpp/examples/paper`.

## How to Run

1.  **Build C++ Benchmarks**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Run C++ Simulations**:
    Execute the compiled C++ simulation binary:
    ```bash
    ./bin/comparison_dabm_ode_hybrid
    ./bin/abm_ide_lct_ode
    ```

4.  **Generate Plots**:
    Use the Python scripts for visualization by setting the simulation output directory and executing e.g.:
    ```bash
    python3 plot_hybrid_application.py
    ```
    Note, that this script uses the general configuration file 'plottings_settins.py'.
