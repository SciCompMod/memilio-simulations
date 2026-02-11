# MEmilio Paper 2026 (Bicker et al.)

This folder contains the code and scripts used to generate the results and figures for the MEmilio paper. A basic introduction on how to setup and run MEmilio's examples is found at https://memilio.readthedocs.io/en/latest/getting_started.html#how-to-use-memilio. The initial installation time on an individual computer should take several minutes.

## Structure

The contributions are organized by figure and (if necessary) additional subfolders for the panels (e.g., `Fig6_a/`).

### Versioning

To ensure reproducibility, each subfolder with simulations contains a `git_tag.cmake` file. This file specifies the exact commit hash of the [MEmilio repository](https://github.com/SciCompMod/memilio) required to run the code in that folder.

### Usage

1.  **Check the `git_tag.cmake`**: Look into the specific figure folder to find the required MEmilio version.
2.  **Read the Folder README**: Each subfolder has its own `README.md` with specific instructions on how to build the simulation and generate the plots.
