#!/bin/bash
#SBATCH --job-name=lct-performance
#SBATCH --output=lct-%A.out
#SBATCH --error=lct-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --exclude="be-cpu05, be-gpu01"
#SBATCH --time=5-0:00:00

## This script can be used to monitor runtimes for the lct model using the file lct_runtime.cpp.
## The command "sbatch get_runtimes_lct.sh ./bin/lct_runtime" should be run in this folder 
## to have valid folder instructions.

num_runs=100
num_warm_up_runs=10
# Use 1 to measure run times for an adaptive solver, 0 for fixed step sizes.
use_adaptive_solver=0
echo Running $1 on node $SLURM_JOB_NODELIST with $num_warm_up_runs warm up runs and $num_runs runs.

# Load module with likwid included
module purge
module load PrgEnv/gcc13-openmpi
# Remove previous build folders and rebuild
rm -rf build
cd ../ 
rm -rf build
mkdir build
cd build/
cmake .. -DBUILD_2024_Ploetzke_et_al_Revisiting=ON -DNUM_JOBS_BUILD=32
cd ../2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick/build/
cmake -Wno-dev -DCMAKE_BUILD_TYPE="Release" ..

#for i in {100..2000..100}
for i in {460..600..20}
do  
    cmake -Wno-dev -DNUM_SUBCOMPARTMENTS=$i -DCMAKE_BUILD_TYPE="Release" .
    cmake --build . --target lct_runtime
    # Memory: FLOPS_DP replace by MEM_DP
    echo Run with $i subcompartments and flag MEM_DP.
    srun --cpus-per-task=1 --cpu-bind=cores likwid-perfctr -C 0 -g MEM_DP -m ./$1 $num_runs $num_warm_up_runs $use_adaptive_solver
    echo Run with $i subcompartments and flag FLOPS_DP.
    srun --cpus-per-task=1 --cpu-bind=cores likwid-perfctr -C 0 -g FLOPS_DP -m ./$1 $num_runs $num_warm_up_runs $use_adaptive_solver
    echo Run with $i subcompartments and flag L3CACHE.
    srun --cpus-per-task=1 --cpu-bind=cores likwid-perfctr -C 0 -g L3CACHE -m ./$1 $num_runs $num_warm_up_runs $use_adaptive_solver
    echo Run with $i subcompartments and flag L2CACHE.
    srun --cpus-per-task=1 --cpu-bind=cores likwid-perfctr -C 0 -g L2CACHE -m ./$1 $num_runs $num_warm_up_runs $use_adaptive_solver
    # srun --cpus-per-task=1 --cpu-bind=cores ./$1 $num_runs $num_warm_up_runs $use_adaptive_solver
done
