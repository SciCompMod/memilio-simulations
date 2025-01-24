cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2024_Ploetzke_et_al_Revisiting VERSION 1.0.0)

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
set(MEMILIO_ENABLE_OPENM OFF)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

# Variables as MEMILIO_HAS_HDF5 are not in the current scope and cannot be used.
if(NOT DEFINED NUM_SUBCOMPARTMENTS)
  set(NUM_SUBCOMPARTMENTS "42")
endif()
add_definitions(-DNUM_SUBCOMPARTMENTS=${NUM_SUBCOMPARTMENTS})

add_executable(lct_impact_distribution_assumption lct_impact_distribution_assumption.cpp)
target_link_libraries(lct_impact_distribution_assumption PRIVATE memilio lct_secir)

add_executable(lct_impact_age_resolution lct_impact_age_resolution.cpp)
target_link_libraries(lct_impact_age_resolution PRIVATE memilio lct_secir)

add_executable(lct_covid19_inspired_scenario lct_covid19_inspired_scenario.cpp)
target_link_libraries(lct_covid19_inspired_scenario PRIVATE memilio lct_secir)

if (MEMILIO_ENABLE_OPENMP)
  add_executable(lct_runtime lct_runtime.cpp)
  target_link_libraries(lct_runtime PRIVATE memilio lct_secir)
  # target_compile_options(lct_runtime PRIVATE -O0)
endif()
