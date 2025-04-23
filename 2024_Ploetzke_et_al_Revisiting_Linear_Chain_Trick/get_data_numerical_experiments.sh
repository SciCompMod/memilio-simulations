#!/bin/bash

## This script can be used to get all simulation data for the numerical experiments regarding the impact of the 
## distribution assumption and of the age resolution.
## The files lct_impact_distribution_assumption.cpp and lct_impact_age_resolution.cpp are used.

cd build/
cmake ..

result_dir="../simulation_results/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi
result_dir="../simulation_results/simulation_lct_numerical_experiments/"
if [ ! -d "$result_dir" ]; then
    mkdir "$result_dir"
fi


subdir_dropReff="$result_dir/dropReff/"
if [ ! -d "$subdir_dropReff" ]; then
    mkdir "$subdir_dropReff"
fi
subdir_dropReff40="$result_dir/dropReff40/"
if [ ! -d "$subdir_dropReff40" ]; then
    mkdir "$subdir_dropReff40"
fi
subdir_riseReffTo2short="$result_dir/riseReffTo2short/"
if [ ! -d "$subdir_riseReffTo2short" ]; then
    mkdir "$subdir_riseReffTo2short"
fi
subdir_riseReffTo2_40="$result_dir/riseReffTo2_40/"
if [ ! -d "$subdir_riseReffTo2_40" ]; then
    mkdir "$subdir_riseReffTo2_40"
fi
subdir_riseReffTo2shortTEhalved="$result_dir/riseReffTo2shortTEhalved/"
if [ ! -d "$subdir_riseReffTo2shortTEhalved" ]; then
    mkdir "$subdir_riseReffTo2shortTEhalved"
fi

# Compile with the different numbers of subcompartments and run with different setups.
for num_subcomp in 0 1 3 10 50
do
    cmake -DNUM_SUBCOMPARTMENTS=$num_subcomp -DCMAKE_BUILD_TYPE="Release" .
    cmake --build . --target lct_impact_distribution_assumption

    # First case: Decrease the effective reproduction number at simulation day 2 to 0.5 and simulate for 12 days.
    Reff=0.5
    tReff=2.
    simulation_days=12
    ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_dropReff
    simulation_days=40
    ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_dropReff40

    # Second case: Increase the effective reproduction number at simulation day 2 to 2 and simulate for 12 days.
    Reff=2.
    tReff=2.
    simulation_days=12
    # Additionally save result with division in subcompartments.
    if [ "$num_subcomp" -eq 10 ] || [ "$num_subcomp" -eq 50 ]; then
        ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_riseReffTo2short 1
    fi
    ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_riseReffTo2short
    simulation_days=40
    ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_riseReffTo2_40

    # Third case: Second case but with TimeExposed scaled by 0.5. Exclude Lct Var.
    if [ "$num_subcomp" -ne 0 ]; then 
        scale_TimeExposed=0.5
        tReff=2.
        # Additionally save result with division in subcompartments.
        if [ "$num_subcomp" -eq 50 ]; then
            ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_riseReffTo2shortTEhalved 1 $scale_TimeExposed
        fi
        ./bin/lct_impact_distribution_assumption $Reff $tReff $simulation_days $subdir_riseReffTo2shortTEhalved 0 $scale_TimeExposed
    fi

    # Fourth case: Print final sizes without saving results.
    simulation_days=500
    tReff=0.
    ./bin/lct_impact_distribution_assumption 2 $tReff $simulation_days "" 0 1.0 1
    ./bin/lct_impact_distribution_assumption 4 $tReff $simulation_days "" 0 1.0 1
    ./bin/lct_impact_distribution_assumption 10 $tReff $simulation_days "" 0 1.0 1
done


# Fifth case: Increase the effective reproduction number at simulation day 0 to different values and 
# simulate for 200 days to compare epidemic peaks.
# Also perform simulations with TimeExposed scaled with 0.5 or 2.
# Define and construct relevant folders.
subdir_riseRefflong="$result_dir/riseRefflong/"
if [ ! -d "$subdir_riseRefflong" ]; then
    mkdir "$subdir_riseRefflong"
fi
subdir_riseRefflongTEhalved="$result_dir/riseRefflongTEhalved/"
if [ ! -d "$subdir_riseRefflongTEhalved" ]; then
    mkdir "$subdir_riseRefflongTEhalved"
fi
subdir_riseRefflongTEdoubled="$result_dir/riseRefflongTEdoubled/"
if [ ! -d "$subdir_riseRefflongTEdoubled" ]; then
    mkdir "$subdir_riseRefflongTEdoubled"
fi
simulationdays=200
Reffs=(2 3 4 5 6 7 8 9 10)
tReff=0.
for num_subcomp in 0 1 2 3 4 5 6 7 8 9 10 50
do
    cmake -DNUM_SUBCOMPARTMENTS=$num_subcomp -DCMAKE_BUILD_TYPE="Release" .
    cmake --build . --target lct_impact_distribution_assumption
    for index in {0..8}
    do
        ./bin/lct_impact_distribution_assumption ${Reffs[index]} $tReff ${simulationdays} $subdir_riseRefflong
        if [ "$num_subcomp" -ne 0 ]; then 
            ./bin/lct_impact_distribution_assumption ${Reffs[index]} $tReff ${simulationdays} $subdir_riseRefflongTEhalved 0 0.5
            ./bin/lct_impact_distribution_assumption ${Reffs[index]} $tReff ${simulationdays} $subdir_riseRefflongTEdoubled 0 2.0
        fi
    done
done

# Sixth case: Simulation for the impact of age resolution.
# 40 days.
subdir_age_resolution_short="$result_dir/age_resolution_short/"
if [ ! -d "$subdir_age_resolution_short" ]; then
    mkdir "$subdir_age_resolution_short"
fi
cmake --build . --target lct_impact_age_resolution 
contact_data_dir="_deps/memilio-src/data/contacts/"
tmax=40
./bin/lct_impact_age_resolution $contact_data_dir $subdir_age_resolution_short $tmax
# 200 days.
subdir_age_resolution_long="$result_dir/age_resolution_long/"
if [ ! -d "$subdir_age_resolution_long" ]; then
    mkdir "$subdir_age_resolution_long"
fi
cmake --build . --target lct_impact_age_resolution 
contact_data_dir="_deps/memilio-src/data/contacts/"
tmax=200
./bin/lct_impact_age_resolution $contact_data_dir $subdir_age_resolution_long $tmax
