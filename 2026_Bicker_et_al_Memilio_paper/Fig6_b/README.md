# Figure 6b, f: Ensemble simulations of the ODE, IDE and LCT model

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for the plots.

## Files

- **Simulations**:
    - `ode_runtime.cpp`: Scaling of the Graph-ODE simulation using with respect to the number of nodes.
    - `ide_runtime.cpp`: Scaling of the Graph-IDE simulation using with respect to the number of nodes.
    - `lct_runtime.cpp`: Scaling of the Graph-LCT simulation using with respect to the number of nodes.
- **Shellscripts**: 
    - `get_runtime.sh`: Runs the simulations for a series of numbers of regions.
- **Python scripts**:
    - `runtime_scaling_region_plot.py`: Plotting of the results.

## How to Run

1.  **Install the project**:
    Configure the project using CMake. 

2. **Run the simulation**:
