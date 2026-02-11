cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2026_Bicker_et_al_Memilio_paper-Fig6_d VERSION 1.0.0)

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

# Command used to patch differences from the developement branch to GIT_TAG_MEMILIO
set(memilio_patch git apply ${CMAKE_CURRENT_SOURCE_DIR}/paper-compare-abm-ide-lct-ode.patch)

FetchContent_Declare(
  memilio
  GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
  GIT_TAG ${GIT_TAG_MEMILIO}
  # Apply the patch, but only once
  PATCH_COMMAND ${memilio_patch}
  UPDATE_DISCONNECTED 1
)

FetchContent_MakeAvailable(memilio)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)
set(MEMILIO_ENABLE_OPENMP ON)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

# Set C++ version for simulation files
set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED "20")

# Just build example if MEMILIO_ENABLE_OPENMP is explicitly set to on.
if(MEMILIO_ENABLE_OPENMP)
  add_executable(scaling_dabm_ode_hybrid scaling_dabm_ode_hybrid.cpp)
  target_link_libraries(scaling_dabm_ode_hybrid PRIVATE memilio hybrid ode_secir d_abm)
endif()
