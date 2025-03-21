# Appropriate relaxation of non-pharmaceutical interventions minimizes the risk of a resurgence in SARS-CoV-2 infections in spite of the Delta variant #

This directory contains the simulation file [vaccination_sarscov2_delta_germany](vaccination_sarscov2_delta_germany.cpp) related to the paper

- _Wadim Koslow, Martin J. Kühn, Sebastian Binder, Margrit Klitz, Daniel Abele, Achim Basermann, Michael Meyer-Hermann. Appropriate relaxation of non-pharmaceutical interventions minimizes the risk of a resurgence in SARS-CoV-2 infections in spite of the Delta variant, PLOS Computational Biology (2022)._ 
https://doi.org/10.1016/j.mbs.2021.108648.


Extending the model of [2020_npis_sarscov2_wildtype_germany](../2020_npis_sarscov2_wildtype_germany) by vaccination and reinfection and considering the effect of vaccination in combination with the lifting of NPIs during the arrival of Delta.

## Data availability
The data which is necessary to run these simulations is generated by the MEmilio epidata package.
This can be done by calling the particular python routines manually or by running the [data_generation.sh](data_generation.sh) shell script.
The shell script assumes the existence of a python environment and only the path activation function of the environment 
must be given to the script. The path can either be set in the script or by calling the script directly with the option
"-PATH_ENV YOUR_PATH_TO_VIRTUAL_ENV/ACTIVATION_EXEC".
The script needs to be executed from the current folder directly by `./data_generation.sh`.
The CMake project has to be build beforehand such that a `build/` folder with the corresponding downloaded MEmilio exists.

## Requirements
We require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)).
