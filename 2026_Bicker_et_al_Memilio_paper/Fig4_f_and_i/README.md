# Figure 4f, i: Different intervention strategies on global and local level in Germany

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for the creation of the table.

## Files

- **Simulation**:
    - `graph_germany_nuts3_ode.cpp`: Simulation of different intervention strategies for Germany on district level (Fig. 4f).
- **Python scripts**:
    - `nuts3_process_contacts.py`: Calculation of the average number of realized contacts and outcomes for infected, severe, and critical cases for each scenario (Fig. 4i).

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake).

2. **Download the required data**:
    Download the data for Germany by running 
    ```bash
    cd 2026_Bicker_et_al_Memilio_paper/Fig4_f_and_i
    source build/venv/bin/activate
    python build/_deps/memilio-src/pycode/memilio-epidata/memilio/epidata/getSimulationData.py -o "build/_deps/memilio-src/data"
    ```

3. **Run the simulation**:
    Run the example graph_germany_nuts3_ode.cpp with one of the four scenarios (0-4).
    
    For example, for the scenario of no NPIs in place:
    ```bash
    ./build/bin/sim_graph_germany_nuts3_ode -TestCase 0
    ```

    The possible scenarios are:

    0: "Open" - No NPIs in place

    1: "Same" - Continuation of the fitted NPI strengths

    2: "Lockdown" - Emplacing 1.2 * fitted NPI strengths

    3: "Dynamic" - Dynamic NPIs that are triggered once a threshold is reached.

4. **Create the table of Fig. 4i**
    To create the table, save the terminal output of step 2 in a file named `results/Output_<case>.txt` where \<case\> should be replaced by the scenario name.
    Then run the file with 

    ```bash
    source build/venv/bin/activate
    python nuts3_preprocess_contacts.py
    ```