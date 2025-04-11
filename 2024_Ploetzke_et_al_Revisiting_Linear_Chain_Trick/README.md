# Revisiting the Linear Chain Trick in epidemiological models: Implications of underlying assumptions for numerical solutions #

This directory contains all files related to the paper

- _Lena Plötzke, Anna Wendler, René Schmieding, Martin J. Kühn. Revisiting the Linear Chain Trick in epidemiological models: Implications of underlying assumptions for numerical solutions,  (2024)._ 
https://doi.org/10.48550/arXiv.2412.09140.

## Requirements
We require the libraries `JsonCpp` and `HDF5` for running the simulations (these are optional for the MEmilio project, see [MEmilio cpp README](https://github.com/SciCompMod/memilio/blob/main/cpp/README.md)). For the runtime measurements, `HDF5` is not required but `OpenMP`.

The memilio.epidata package needs to be installed for the python plot scripts and the data download. 
Have a look at the [pycode README](../../../pycode/README.rst) and the [memilio-epidata README](../../../pycode/memilio-epidata/README.rst) for instructions how to install the package.

## Overview
Below is an overview of the files and the paper sections they belong to.

- Section 4.2: The file [lct_impact_distribution_assumption](lct_impact_distribution_assumption.cpp) provides the functionality to run simulations to assess the impact of the distribution assumption by choosing different numbers of subcompartments. The dynamics at change points and epidemic peaks can be examined using this script. The population is not divided into age groups for these experiments. All simulation results are created and saved in `.h5` files when the shell script [get_data_numerical_experiments](get_data_numerical_experiments.sh) is executed. The visualizations of these simulation results in the paper were created using the python script [plot_numerical_experiments](plot_numerical_experiments.py).

- Section 4.3: With the file [lct_impact_age_resolution](lct_impact_age_resolution.cpp), one can run simulations to assess the impact of including an age resolution into the model. The simulation results are created and saved together with the results for section 4.2 with the shellscript [get_data_numerical_experiments](get_data_numerical_experiments.sh). The visualizations are also created with [plot_numerical_experiments](plot_numerical_experiments.py).

- Section 4.4: Run time measurements are possible with the file [lct_runtime](lct_runtime.cpp). `OpenMP` is used to measure the run times. The target `lct_runtime` is only built if the option `MEMILIO_ENABLE_OPENMP` is enabled. The Slurm script [get_runtimes_lct](get_runtimes_lct.sh) can be used to define a job to measure the run time for different numbers of subcompartments. This script can be easily adapted for the use of an adaptive solver. To use the optimization flag `-O0`, uncomment the suitable line in the [CMakeLists file](CMakeLists.txt). Visualizations of the run times in the paper were created using the python script [plot_runtimes_lct](plot_runtimes_lct.py).

- Section 4.5: A COVID-19 inspired scenario in Germany in 2020 is defined in the file [lct_covid19_inspired_scenario](lct_covid19_inspired_scenario.cpp). The simulation results are created and saved with the shell script [get_data_covid19_inspired](get_data_covid19_inspired.sh). 
The simulation is initialized using reported data, which has to be downloaded beforehand. Please execute the file [download_infection_data](download_data.py) for this purpose.
The visualizations of the simulation results in the paper were created using the python script [plot_covid19_inspired](plot_covid19_inspired.py).

- Figure 2 and Figure 12: These figures are not based on simulation results. Figure 2 contains a visualization of the density and the survival function of Erlang distributions with different parameter choices. Figure 12 shows the age-resolved contact pattern for Germany. Both plots are created using [plot_details](plot_details.py).


For most of the above `.cpp` files, the number of subcompartments used in the LCT models for all compartments can be controlled via the preprocessor macro NUM_SUBCOMPARTMENTS. Have a look at the files for further documentation or the shell scripts for the usage. 
