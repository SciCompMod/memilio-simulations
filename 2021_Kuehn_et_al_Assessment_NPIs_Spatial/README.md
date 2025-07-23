# Assessment of effective mitigation and prediction of the spread of SARS-CoV-2 in Germany using demographic information and spatial resolution #

This directory contains the simulation file related to the paper

- _Martin J. Kühn, Daniel Abele, Tanmay Mitra, Wadim Koslow, Majid Abedi, Kathrin Rack, Martin Siggel, Sahamoddin Khailaie, Margrit Klitz, Sebastian Binder, Luca Spataro, Jonas Gilg, Jan Kleinert, Matthias Häberle, Lena Plötzke, Christoph D. Spinner, Melanie Stecher, Xiao Xiang Zhu, Achim Basermann, Michael Meyer-Hermann. Assessment of effective mitigation and prediction of the spread of
SARS-CoV-2 in Germany using demographic information and spatial
resolution, Mathematical Biosciences (2021)._ 
https://doi.org/10.1016/j.mbs.2021.108648.


We provide a spatially resolved simulation [npis_sarscov2_wildtype_germany](npis_sarscov2_wildtype_germany.cpp). 
The graph-ODE (or metapopulation) model uses one ODE model for each county and realizes inter-county mobility via a graph approach.
The focus of this simulation is on a SECIR model with parameters for Sars-CoV-2 wild type variant and
implementing static nonpharmaceutical interventions (NPIs) as well as dynamic NPIs. Dynamic NPIs
get into play once predefined incidence thresholds (50 and 200) are exceeded.
The Parameters `TimeExposed` and `TimeInfectedNoSymptoms` can be derived from the serial interval and incubation period, as delineated by Khailaie and Mitra et al. (https://doi.org/10.1186/s12916-020-01884-4).

## Additional Scripts

This folder contains additional utility scripts for data generation and result visualization:

- **`data_generation.sh`**: Script to generate input data for the simulation using the Memilio Epidata package. This script requires a Python virtual environment and downloads data from external sources.

- **`plot_results_odesecir.ipynb`**: Jupyter notebook containing plotting scripts to recreate the visualizations from the paper. This notebook includes functions to plot simulation results across different age groups and compartments.

**Disclaimer**: These scripts are provided as reference only, without any guarantee that they will work correctly outside of the environment they were written in. Please refer to the main repository disclaimer for more information about potential compatibility issues with external data sources.

## Requirements
We require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)).
