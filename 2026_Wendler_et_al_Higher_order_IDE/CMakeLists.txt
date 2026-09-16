cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2026_Wendler_et_al_Higher_order VERSION 1.0.0)

# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Executables should be stored in the build/bin/ folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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
  SOURCE_SUBDIR cpp
)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)

FetchContent_MakeAvailable(memilio)

# Variables as MEMILIO_HAS_HDF5 are not in the current scope and cannot be used.

add_executable(init_conditions_exp init_conditions_exp.cpp)
target_link_libraries(init_conditions_exp PRIVATE memilio ode_sir ide_sir)

add_executable(init_conditions_erlang init_conditions_erlang.cpp)
target_link_libraries(init_conditions_erlang PRIVATE memilio lct_sir ide_sir)

add_executable(convergence_constant_contacts convergence_constant_contacts.cpp)
target_link_libraries(convergence_constant_contacts PRIVATE memilio ode_sir ide_sir)

add_executable(convergence_nonconstant_contacts convergence_nonconstant_contacts.cpp)
target_link_libraries(convergence_nonconstant_contacts PRIVATE memilio ode_sir ide_sir)

add_executable(runtime_smoother runtime_smoother.cpp)
target_link_libraries(runtime_smoother PRIVATE memilio)