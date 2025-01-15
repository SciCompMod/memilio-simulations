# 2020_npis_sarscov2_wildtype_germany #

This folder provides a spatially resolved simulation. 
The graph-ODE (or metapopulation) model uses one ODE model for each county and realizes inter-county mobility via a graph approach.
The focus of this simulation is on a SECIR model with parameters for Sars-CoV-2 wild type variant and
implementing static nonpharmaceutical interventions (NPIs) as well as dynamic NPIs. Dynamic NPIs
get into play once predefined incidence thresholds (50 and 200) are exceeded.
The Parameters `TimeExposed` and `TimeInfectedNoSymptoms` can be derived from the serial interval and incubation period, as delineated by Khailaie and Mitra et al. (https://doi.org/10.1186/s12916-020-01884-4).

# Requirements
We require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)).
