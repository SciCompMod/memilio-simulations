# MEmilio-simulations

This Repository contains simulations using the library [MEmilio](https://github.com/SciCompMod/memilio).
Each simulation is in a separate folder, containing the version of MEmilio used to create the simulations.
This way, simulation results should be easy to recreate.

## Disclaimer

Additional scripts used for the creation of published simulation results may be provided in each folder, for example for plotting, gathering data or pre-/post-processing. These scripts are provided as a reference, without any guarantee that they will work correctly outside of the environment they were written in. Do not use them without prior inspection.

In addition, some scripts, like the memilio-epidata tools in the pycode folder of MEmilio, depend on external data sources. Expired links will be updated or replaced in the main branch of the MEmilio repository, but these updates are not brought to the specific versions of MEmilio used by the simulations here. You may try to use links or scripts from the [current version of MEmilio](https://github.com/SciCompMod/memilio), but be aware that they may be incompatible due to structural changes in the data sources or code.

## Configuring using CMake

To configure and build a simulation, choose one option from the list below and run:

```bash
mkdir build && cd build
cmake .. -D<OPTION>=ON
```

Please check the README of the subdirectories for details about the simulations.
The following `<OPTION>`s can be set:

- `BUILD_2021_Kuehn_et_al_Assessment`
- `BUILD_2022_Koslow_et_al_Appropriate`
- `BUILD_2024_Ploetzke_et_al_Revisiting`
- `BUILD_2024_Wendler_et_al_Nonstandard`
- `BUILD_munich_graph_sim`
- `BUILD_2025_Kerkmann_Korf_et_al_Testing`
- Figures for the MEmilio paper (2026_Bicker_et_al_Memilio_paper). Note that each Figure has its own subdirectory.
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_b`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_c`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_d_and_e`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_f_g_and_i`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_h`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig5`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_a`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_b_and_f`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_c_and_f`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_d`
  - `BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_e`

If you do not set an option, no simulation will be built. After a simulation has been built, you can prevent cmake from rebuilding it by setting `cmake .. -D<OPTION>=OFF`, although rebuilding should only take a few seconds. You can also set multiple options at once by adding more pairs `-D<OPTION>=<VALUE>` to the end, with `<VALUE>` being either `ON` or `OFF`.

A build folder with the correct MEmilio version will be created in each subdirectory.
You can run a simulation with e.g.:

```bash
../2021_Kuehn_et_al_Assessment_NPIs_Spatial/build/bin/npis_sarscov2_wildtype_germany
```

Furthermore, you can specify the maximum number of concurrent processes to use when building by setting the variable `NUM_JOBS_BUILD`. Using a value higher than `1` could speed up the build process. The default of this variable is `1`. With this specification, the build process should work for every hardware and operating system.

**If the build fails, you may be missing requirements/dependencies!**

## Requirements

For most simulations, we require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)). The git repo of the library `JsonCpp` is still bundled with MEmilio while `HDF5` is not. Please have a look at the README in a specific folder for further specifications.

## Information for Developers

If you want to create a new folder, e.g. for the files of a new paper, you should follow the steps below:

- Create a folder with a descriptive name, referenced in the following as `<FolderName>`.
  Typically, we use `<Year>_<FirstAuthor>_et_al_<FirstWordTitle>_<AdditionalWords>`.
  The name should clearly indicate the paper to which the folder is related.
- Put all related files in it.
- In a file `git_tag.cmake`, define a git tag for the MEmilio version used (can be a commit hash or a branch name).
  This is done using the line

```bash
set(GIT_TAG_MEMILIO <commit_hash>)
```

- Add a `CMakeLists.txt` and define a unique project name (typically `memilio-simulations-<Year>_<FirstAuthor>_et_al_<FirstWordTitle>`). Additionally add the code block

```bash
# Executables should be stored in the build/bin/ folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
```

for the stated reason.
We need a code block to download MEmilio with the version defined in `git_tag.cmake`:

```bash
# Download MEmilio with the git tag defined in git_tag.cmake.
include(${CMAKE_SOURCE_DIR}/git_tag.cmake)
# If git tag is not set, throw error.
if(NOT DEFINED GIT_TAG_MEMILIO)
    message(FATAL_ERROR "GIT_TAG_MEMILIO is not defined. Please make sure the git_tag.cmake file is correct.")
endif()

# FetchContent to fetch the MEmilio library in the correct version.
include(FetchContent)
    
FetchContent_Declare(
memilio
GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
GIT_TAG ${GIT_TAG_MEMILIO}
)

FetchContent_MakeAvailable(memilio)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})
```

Finally, create compilation targets for the `.cpp`-files, and link all required libraries, like memilio or the model libraries used by the simulation.

- In the global `CMakeLists.txt`, add an option `BUILD_<Year>_<FirstAuthor>_et_al_<FirstWordTitle>` for your new content and set the default option to `OFF`:

```bash
option(BUILD_<Year>_<FirstAuthor>_et_al_<FirstWordTitle> "Build simulations from folder <FolderName>." OFF)
```

Please also add the option to the current `README.md`.

- Additionally, in the global `CMakeLists.txt`, add the commands to build your files using the local `CMakeLists.txt`:

```bash
if(BUILD_<Year>_<FirstAuthor>_et_al_<FirstWordTitle>)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/<FolderName>/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/<FolderName>")
  endif()
  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/<FolderName>/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/<FolderName>/build")
endif()
```

- Clearly state all requirements in your local `README.md` (e.g. hdf5) as we cannot access the MEmilio variables MEMILIO_HAS... to control the requirements in CMake.

### Making and applying patches

If the simulation is based on a developement branch with changes that are not merged into the main branch of MEmilio,
you should create a patch file relative to a commit on the main branch (preferably the most recent one).
You can create a patch file from within the main MEmilio repository by using

```bash
git diff --patch --ignore-space-change --minimal --output=<MyBranch>.patch $(git merge-base <MyBranch> main)..<MyBranch>
```

where "MyBranch" is replaced by the name of the developement branch.
You can restrict the patch to certain directories (like "cpp/memilio") by adding paths to the end of this command.
The commit hash obtained from
`git merge-base <MyBranch> main`
must then also be used as GIT_TAG_MEMILIO.

Then, move the patch file to `<FolderName>`, and add the following two items to the CMakeLists.txt there:

First, set the patch command above the `FetchContent_Declare(memilio ... )`:
```bash
set(memilio_patch git apply ${CMAKE_CURRENT_SOURCE_DIR}/<MyBranch>.patch)
```

Second, add the following two items to the end of the arguments in `FetchContent_Declare(memilio ... )`:
```bash
FetchContent_Declare(
  memilio
  ...
  PATCH_COMMAND ${memilio_patch}
  UPDATE_DISCONNECTED 1
)
```
The first line allows for applying the patch when memilio is included, the second makes sure this only happens once. 
