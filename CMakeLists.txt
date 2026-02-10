cmake_minimum_required(VERSION 3.13)

project(memilio-simulations VERSION 1.0.0)

if(NOT DEFINED NUM_JOBS_BUILD)
  set(NUM_JOBS_BUILD "1")
endif()

option(BUILD_2021_Kuehn_et_al_Assessment "Build simulations from folder 2021_Kuehn_et_al_Assessment_NPIs_Spatial." OFF)
option(BUILD_2022_Koslow_et_al_Appropriate "Build simulations from folder 2022_Koslow_et_al_Appropriate_Relaxation_NPIs." OFF)
option(BUILD_2024_Ploetzke_et_al_Revisiting "Build simulations from folder 2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick." OFF)
option(BUILD_2024_Wendler_et_al_Nonstandard "Build simulations from folder 2024_Wendler_et_al_Nonstandard_Numerical_Scheme_IDE." OFF)
option(BUILD_munich_graph_sim "Build simulations from folder munich_graph_sim." OFF)
option(BUILD_2025_Kerkmann_Korf_et_al_Testing "Build simulations from folder ABM_TEST_BS_simulation." OFF)
option(BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_d "Build simulations from folder 2026_Bicker_et_al_Memilio_paper for Figure 4d." OFF)
option(BUILD_2026_Bicker_et_al_Memilio_paper-Fig5 "Build simulations from folder 2026_Bicker_et_al_Memilio_paper for Figure 5." OFF)
option(BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_a "Build simulations from folder 2026_Bicker_et_al_Memilio_paper for Figure 6a." OFF)
option(BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_d "Build simulations from folder 2026_Bicker_et_al_Memilio_paper for Figure 6d." OFF)

# We require the library hdf5 for all targets.
find_package(HDF5 REQUIRED)

if(BUILD_2021_Kuehn_et_al_Assessment)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2021_Kuehn_et_al_Assessment_NPIs_Spatial/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2021_Kuehn_et_al_Assessment_NPIs_Spatial")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2021_Kuehn_et_al_Assessment_NPIs_Spatial/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2021_Kuehn_et_al_Assessment_NPIs_Spatial/build")
endif()

if(BUILD_2022_Koslow_et_al_Appropriate)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2022_Koslow_et_al_Appropriate_Relaxation_NPIs/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2022_Koslow_et_al_Appropriate_Relaxation_NPIs")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2022_Koslow_et_al_Appropriate_Relaxation_NPIs/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2022_Koslow_et_al_Appropriate_Relaxation_NPIs/build")
endif()

if(BUILD_2024_Ploetzke_et_al_Revisiting)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Ploetzke_et_al_Revisiting_Linear_Chain_Trick/build")
endif()

if(BUILD_2024_Wendler_et_al_Nonstandard)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2024_Wendler_et_al_Nonstandard_Numerical_Scheme_IDE/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Wendler_et_al_Nonstandard_Numerical_Scheme_IDE")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Wendler_et_al_Nonstandard_Numerical_Scheme_IDE/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2024_Wendler_et_al_Nonstandard_Numerical_Scheme_IDE/build")
endif()

if(BUILD_munich_graph_sim)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/munich_graph_sim/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/munich_graph_sim")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/munich_graph_sim/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/munich_graph_sim/build")
endif()

if(BUILD_2025_Kerkmann_Korf_et_al_Testing)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2025_Kerkmann_Korf_et_al_agent_based_model_testing/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2025_Kerkmann_Korf_et_al_agent_based_model_testing")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2025_Kerkmann_Korf_et_al_agent_based_model_testing/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2025_Kerkmann_Korf_et_al_agent_based_model_testing/build")
endif()

if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_d_and_e)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_d_and_e/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_d_and_e")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_d_and_e/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_d_and_e/build")
endif()

if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig4_f_and_i)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_f_and_i/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_f_and_i")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_f_and_i/build")
  execute_process(COMMAND cmake "--build" "."  "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig4_f_and_i/build")
endif()

if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig5)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig5/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig5")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig5/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig5/build")
endif()

if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_a)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_a/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_a")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_a/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_a/build")
endif()

if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_b)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_b/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_b")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_b/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_b/build")
endif()


if(BUILD_2026_Bicker_et_al_Memilio_paper-Fig6_d)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_d/build")
    execute_process(COMMAND mkdir "build/" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_d")
  endif()

  execute_process(COMMAND cmake ".." WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_d/build")
  execute_process(COMMAND cmake "--build" "." "-j${NUM_JOBS_BUILD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/2026_Bicker_et_al_Memilio_paper/Fig6_d/build")
endif()
