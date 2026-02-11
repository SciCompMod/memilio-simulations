#!/bin/bash
#SBATCH --job-name=ode-ensemble
#SBATCH --output=ode-ensemble%A.out
#SBATCH --error=ode-ensemble%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --nodelist="be-cpu04, be-cpu02, be-cpu03"
#SBATCH --time=1-0:00:00

# Run this script from build folder with downloaded data in repository
echo Running on node $SLURM_JOB_NODELIST.


# Define parameters used as command line arguments.
num_runs=1024

for num_mpi in 1 2 4 8 16 32 64 128 168
do
    # Simulation for 01/10/2020.
    mpirun -n $num_mpi ./build/bin/sim_ode_ensemble_runs -NumberEnsembleRuns $num_runs
done
