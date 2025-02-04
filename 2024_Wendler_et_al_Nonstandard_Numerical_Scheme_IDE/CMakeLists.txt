cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2024_Wendler_et_al_Nonstandard VERSION 1.0.0)

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
)

FetchContent_MakeAvailable(memilio)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

# Variables as MEMILIO_HAS_HDF5 are not in the current scope and cannot be used.

add_executable(ide_convergence_rate ide_convergence_rate.cpp)
target_link_libraries(ide_convergence_rate PRIVATE memilio ode_secir ide_secir Boost::filesystem)

add_executable(compute_parameters compute_parameters.cpp)

add_executable(ide_changepoints ide_changepoints.cpp)
target_link_libraries(ide_changepoints PRIVATE memilio ode_secir ide_secir Boost::filesystem)

add_executable(ide_covid_inspired_scenario ide_covid_inspired_scenario.cpp)
target_link_libraries(ide_covid_inspired_scenario PRIVATE memilio ode_secir ide_secir Boost::filesystem)

