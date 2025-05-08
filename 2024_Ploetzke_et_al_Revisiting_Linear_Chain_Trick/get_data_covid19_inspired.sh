#!/bin/bash

## This script can be used to get all simulation data using the lct_covid19_inspired_scenario.cpp file.
## It is necessary to download RKI and DIVI data beforehand. For more information see the README.

cd build/
cmake ..

infection_data_dir="../data"
contact_data_dir="_deps/memilio-src/data/contacts/"
result_dir="../simulation_results/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi
result_dir="../simulation_results/simulation_lct_covid19/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi

# Define parameters used as command line arguments.
year=2020
RelativeTransmissionNoSymptoms=1.
RiskOfInfectionFromSymptomatic=0.3
month_oct=10
day_oct=1
scale_contacts_oct=0.6072 #28.9.: 0.4060 #29.9.: 0.6048 #30.9.: 0.6318 #2.10.: 0.6422 #3.10.: 0.6580 #4.10.:0.5565
 #1.10.: 0.6537 #1.10._ma7:0.6072
npi_size_oct=0.3
scale_confirmed_cases_oct=1.2

num_runs=2

# Compile with different numbers of subcompartments and run simulations.
# Additionally: Setup with numbers of subcompartments so that each corresponds to the approximate stay time in the compartment.
# This is done by setting the makro NUM_SUBCOMPARTMENTS to zero.
for num_subcomp in 39
do
    # cmake -DNUM_SUBCOMPARTMENTS=$num_subcomp -DCMAKE_BUILD_TYPE="Release" .
    cmake --build . --target lct_covid19_inspired_scenario

    # Simulation for 01/10/2020.
    ./bin/lct_covid19_inspired_scenario $infection_data_dir $contact_data_dir $result_dir $year $month_oct $day_oct $RelativeTransmissionNoSymptoms $RiskOfInfectionFromSymptomatic $scale_contacts_oct $scale_confirmed_cases_oct $npi_size_oct $num_runs
done
