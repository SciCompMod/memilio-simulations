# Figure 4g, h: Optimization of NPI strengths for Germany

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for the creation of the table.

## Files

- **Simulation**:
    - `opt_scenario.cpp`: Optimization of NPI strengths in Germany (Fig. 4g, h).
- **Python scripts**:
    - `plot_results.py`: Creation of the plots.

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Download the required data**:
    Download the data for Germany by running 
    ```bash
    cd 2026_Bicker_et_al_Memilio_paper/Fig4_g_and_h
    source build/venv/bin/activate
    python build/_deps/memilio-src/pycode/memilio-epidata/memilio/epidata/getSimulationData.py -o "build/_deps/memilio-src/data"
    ```

3. **Run the simulation**:
    Run the example graph_germany_nuts3_optimal_control with
    
    For example, for the scenario of no NPIs in place:
    ```bash
    ./build/bin/graph_germany_nuts3_optimal_control
    ```

4. **Create the plots**
    To create the plots, run

    ```bash
    python plot_results.py
    ```