#!/bin/bash
#SBATCH --job-name=lct-ensemble
#SBATCH --output=lct-%A.out
#SBATCH --error=lct-%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --nodelist="be-cpu02, be-cpu03, be-cpu04"
#SBATCH --time=5-0:00:00

## This script can be used to get all simulation data using the lct_covid19_inspired_scenario.cpp file.
## It is necessary to download RKI and DIVI data beforehand. For more information see the README.
echo Running on node $SLURM_JOB_NODELIST.

module purge
module load PrgEnv/gcc12-openmpi

# rm -rf build
# cd ../ 
# rm -rf build
# mkdir build
# cd build/
# cmake .. -DBUILD_2024_Ploetzke_et_al_Revisiting=ON -DNUM_JOBS_BUILD=32
# cd ../2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick/build/
cd build/

num_subcomp=0
cmake -Wno-dev -DNUM_SUBCOMPARTMENTS=$num_subcomp -DCMAKE_BUILD_TYPE="Release" .
cmake --build . --target lct_covid19_inspired_scenario


infection_data_dir="../data"
contact_data_dir="_deps/memilio-src/data/contacts/"
result_dir="../simulation_results/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi
result_dir="../simulation_results/simulation_lct_covid19_ensemble/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi

# Define parameters used as command line arguments.
year=2020
RelativeTransmissionNoSymptoms=1.
RiskOfInfectionFromSymptomatic=0.3
month_oct=10
day_oct=1
scale_contacts_oct=0.6072
npi_size_oct=0.3
scale_confirmed_cases_oct=1.2
num_runs=1024 #16384

# Compile with different numbers of subcompartments and run simulations.
# Additionally: Setup with numbers of subcompartments so that each corresponds to the approximate stay time in the compartment.
# This is done by setting the makro NUM_SUBCOMPARTMENTS to zero.
for num_mpi in 1 2 4 8 16 32 64 128 168
do
    # Simulation for 01/10/2020.
    mpirun -n $num_mpi ./bin/lct_covid19_inspired_scenario $infection_data_dir $contact_data_dir $result_dir $year $month_oct $day_oct $RelativeTransmissionNoSymptoms $RiskOfInfectionFromSymptomatic $scale_contacts_oct $scale_confirmed_cases_oct $npi_size_oct $num_runs
done
