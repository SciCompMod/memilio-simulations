# MEmilio-simulations
This Repository contains simulations using the library [MEmilio](https://github.com/SciCompMod/memilio).

## Configuring using CMake

To configure with default options (no simulations will be built):
```bash
mkdir build && cd build
cmake ..
```

You can specify which simulation folders to build by specifying options via `cmake .. -D<OPTION>=<VALUE>`.
All options can be set to ON or OFF and the default value is always OFF. Please check the README of the subdirectories for details about the simulations.
The following options can be set:
- `BUILD_2020_npis_sarscov2_wildtype_germany`
- `BUILD_2021_vaccination_sarscov2_delta_germany`
- `BUILD_munich_graph_sim`
- `BUILD_2024_Ploetzke_Linear_Chain_Trick`.

A build folder with the correct MEmilio version will be created in each subdirectory. 
You can run a simulation with e.g.:
```bash
./2020_npis_sarscov2_wildtype_germany/build/2020_npis_wildtype
```

## Requirements
For most simulations, we require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)). Please have a look at the folder READMEs for further specifications.

## Information for Developer
