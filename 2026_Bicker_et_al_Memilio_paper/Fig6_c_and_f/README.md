# Figure 6e:ABM scaling

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for plotting results and the Python benchmark script.
- **R**: See [src/dependencies](https://github.com/kilianvolmer/OpenCOVID/blob/main/src/dependencies.R) for required packages.

## Files

- **C++/Python Simulations**:
    - `abm_benchmark_strong.cpp`: Main simulation file for strong sclaing with multiple nodes as well as one node.
    - `abm_benchmark_scaling.cpp`: Main simulation file for runtime/population scaling for the MEmilio-ABM with 1/4/16 cores.
    - `covasim_benchmark.py`: Main simulation file for runtime/population scaling for the Covasim-ABM.
    - https://github.com/kilianvolmer/OpenCOVID/tree/main : Repository where the exeuction of the OpenCOVID was handled.
- **Plotting**:
    - `benchmark_results.py`: Generates Fig. 6d from outputs of scaling_dabm_ode_hybrid.cpp.

## How to Run

1.  **Build C++ Benchmarks**:
    Configure and build the project using CMake, ensuring the `memilio` library is checked out at the commit specified in `git_tag.cmake`. For details, see the Memilio documentation.

2. **Run C++ Simulations**:
    Execute the compiled C++ simulation binary:
    ```bash
    ./bin/abm
    ./bin/abm_parameter_study 
    ```

4.  **Generate Plots**:
    Use the Python scripts for visualization by setting the simulation output directory and executing e.g.:
    ```bash
    python3 plot_hybrid_scaling.py
    ```
    Note, that this script uses the general configuration file 'plottings_settins.py'.

5. **OpenCOVID benchmark**
   Clone [our fork](https://github.com/kilianvolmer/OpenCOVID) of [OpenCOVID](https://github.com/SwissTPH/OpenCOVID) and run the `src/bash_launch.sh` file. Alternatively, you can directly execute the `src/launch.sh` file in R.
