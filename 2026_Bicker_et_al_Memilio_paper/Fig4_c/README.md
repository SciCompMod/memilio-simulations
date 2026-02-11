# Figure 4c: Fits of the ODE SEIRDB-type model for Ebola in Guinea

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for all simulations and plots. See the requirements.txt for the required python packages.

## Files

- **Data**:
    - `cut_early_guinea_conf_cases.csv`: Weekly incidence of influenza cases in Germany between 2016 and 2019.
- **Python Simulations**:
    - `simulation.py`: Simulation of the SEIRDB model.
    - `influenza_plotting.py`: Creation of the plots given samples.
    - `plotting_helpers.py`: Definition of the plotting functions for Fig 4c.
- **Particle Filters and Fitting in Julia**
    - `seirdb_ebola/`: Fitting of the SEIRDB model (mainly in Julia).
    - `SimParticleFilter/`: Utils for the fitting. 

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake). This already installs the necessary packages for the simulations, which takes approximately 15 minutes.

2. **Create the plots given the samples**:
    You can create the plots with
    ```bash
    source build/venv/bin/activate
    python 2026_Bicker_et_al_Memilio_paper/Fig4_c/seirdb_plotting.py
    ```