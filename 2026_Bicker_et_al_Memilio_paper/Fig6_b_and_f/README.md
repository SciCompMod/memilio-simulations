# Figure 6b, f: Ensemble simulations of the ODE, IDE and LCT model

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for the plots.

## Files

- **Simulations**:
    Simulations for Fig. 6b:
    - `ode_runtime.cpp`: Scaling of the Graph-ODE simulation using with respect to the number of nodes.
    - `ide_runtime.cpp`: Scaling of the Graph-IDE simulation using with respect to the number of nodes.
    - `lct_runtime.cpp`: Scaling of the Graph-LCT simulation using with respect to the number of nodes.
    Simulations for Fig. 6f:
    - `ode_runtime.cpp`: Scaling of the Graph-ODE simulation using with respect to the number of nodes.
    - `ide_runtime.cpp`: Scaling of the Graph-IDE simulation using with respect to the number of nodes.
    - `lct_runtime.cpp`: Scaling of the Graph-LCT simulation using with respect to the number of nodes.
- **Shellscripts**: 
    - `get_runtime.sh`: Runs the simulations for a series of numbers of regions.
- **Python scripts**:
    - `runtime_scaling_region_plot.py`: Plotting of the results.

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Get the required data**:
    Download the required data to run the simulations with:
    ```bash
    cd 2026_Bicker_et_al_Memilio_paper/Fig6_b_and_f
    source build/venv/bin/activate
    python build/_deps/memilio-src/pycode/memilio-epidata/memilio/epidata/getSimulationData.py -o "build/_deps/memilio-src/data"
    ```

3. **Run the simulation**:
    Then you can run the simulations for Fig. 6b with 
    ```bash
    ./build/bin/sim_ode_runtime -NumberRuns $num_runs -NumberWarmupRuns $num_warm_up_runs -NumberRegions $i
    ```
    and specify the number of runs, the number of warmup runs and the number of regions.

    With 
    ```bash
    sbatch get_runtime.sh
    ```
    you can run it automatically for 200 to 1000 regions in steps of 50.

    For the parallel scaling, you can run the simulations with 
    ```bash
    mpirun -n $num_mpi ./build/bin/sim_ode_ensemble_runs -NumberEnsembleRuns $num_runs
    ```
    and with 
    ```bash
    sbatch get_ensemble_runs.sh
    ```
    run the simulation with different numbers of threads.

4. **Create the plots**:
    To create the plots, run the plotting script with 
    ```bash
    python runtime_scaling_region_plot.py
    ```
    for the plots in Fig. 6b. The plotting scripts for the strong scaling results can be found [here](../Fig6_c_and_f/).