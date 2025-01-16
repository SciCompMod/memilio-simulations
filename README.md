# MEmilio-simulations
This Repository contains simulations using the library [MEmilio](https://github.com/SciCompMod/memilio).
In this Repo, the simulations are connected with the MEmilio version used to create the simulations. 
This way, the outputs should be easy to recreate. 
All files used to create outputs used e.g. for paper simulations, can be included. 
The scripts do not necessarily have to be portable (e.g. some shellscripts), but they worked locally. Please be aware of tis when using the files.

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
Furthermore, you can specify the maximum number of concurrent processes to use when building by setting the variable `NUM_JOBS_BUILD`. Using a value higher than `1` could speed up the build process. The default of this variable is `1`. With this specification, the build process should work for every hardware and operating system. 

## Requirements
For most simulations, we require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)). Please have a look at the folder READMEs for further specifications.

## Information for Developer
If you want to create a new folder, e.g. for the files for a new paper, then you should follow the steps below:

- Create a folder with a descriptive name.
- Put all related files in it. 
- Add a `CMakeLists.txt` and define a unique project name as well as a part to download MEmilio and create your targets linking the MEmilio targets.
- In a `git_tag.cmake`, define a git tag for the MEmilio version used (can be a commit hash or a branch name).
- In the global `CMakeLists.txt`, add an option for your new content and the commands to build your files using the local `CMakeLists.txt`. Please also add the option to the current `README.md`. 
- Clearly state all requirements in your `README.md` (e.g. hdf5) as we cannot access the MEmilio variables MEMILIO_HAS... to control the requirements in CMake.
