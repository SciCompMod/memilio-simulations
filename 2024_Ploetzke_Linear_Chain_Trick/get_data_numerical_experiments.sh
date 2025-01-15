#!/bin/bash

## This script can be used to get all simulation data for the numerical experiments regarding the impact of the 
## distribution assumption and of the age resolution.
## The files lct_impact_distribution_assumption.cpp and lct_impact_age_resolution.cpp are used.

# Define and construct relevant folders.
if [ ! -d "build/" ]; then
    mkdir "build/"
fi
cd build/
cmake ..

dir="../simulation_results"
if [ ! -d "$dir" ]; then
    mkdir "$dir"
fi
dir="../simulation_results/lct_numerical_experiments"
if [ ! -d "$dir" ]; then
    mkdir "$dir"
fi

subdir_dropReff="$dir/dropReff/"
if [ ! -d "$subdir_dropReff" ]; then
    mkdir "$subdir_dropReff"
fi


# Compile with the different numbers of subcompartments and run with different setups.
for num_subcomp in 1 3 10 50
do
    cmake -DNUM_SUBCOMPARTMENTS=$num_subcomp -DCMAKE_BUILD_TYPE="Release" .
    cmake --build . --target lct_impact_distribution_assumption

    # First case: Decrease the effective reproduction number at simulation day 2 to 0.5 and simulate for 12 days.
    Reff2=0.5
    simulation_days=12
    ./lct_impact_distribution_assumption $Reff2 $simulation_days $subdir_dropReff
done
