cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2024-lct VERSION 1.0.0)

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
if(NOT DEFINED NUM_SUBCOMPARTMENTS)
  set(NUM_SUBCOMPARTMENTS "42")
endif()
add_definitions(-DNUM_SUBCOMPARTMENTS=${NUM_SUBCOMPARTMENTS})
add_executable(lct_impact_distribution_assumption lct_impact_distribution_assumption.cpp)
target_link_libraries(lct_impact_distribution_assumption PRIVATE memilio lct_secir)
