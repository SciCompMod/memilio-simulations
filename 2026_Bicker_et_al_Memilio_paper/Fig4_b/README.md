# Figure 4b: Fits of the SDE SIRS-type model for seasonal influenza in Germany

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for all simulations and plots. See the requirements.txt for the required python packages.

## Files

- **Python Simulations**:
    - `simulation.py`: Simulation of the SDE model.
    - `influenza_plotting.py`: Creation of the plots given samples.
    - `plotting_helpers.py`: Definition of the plotting functions for Fig 4b.
We provide the following files for completeness, although the fitting won't work out of the box and requires setting up the Julia environment first.
- **Particle Filters and Fitting in Julia**
    - `sirs_influenza/`: Fitting of the SIRS model (mainly in Julia).
    - `SimParticleFilter/`: Utils for the fitting. 

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake). This already installs the necessary packages for the simulations, which takes approximately 15 minutes.

2. **Create the plots given the samples**:
    You can create the plots with
    ```bash
    source build/venv/bin/activate
    python 2026_Bicker_et_al_Memilio_paper/Fig4_b/influenza_plotting.py
    ```