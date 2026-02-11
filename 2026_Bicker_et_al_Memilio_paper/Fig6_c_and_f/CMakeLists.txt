cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2026_Bicker_et_al_Memilio_paper-Fig6_c_and_f VERSION 1.0.0)

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
set(memilio_patch git apply --reject ${CMAKE_CURRENT_SOURCE_DIR}/mpi-abm-bm.patch)

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
set(MEMILIO_ENABLE_MPI ON)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

# Set C++ version for simulation files
set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED "20")
# Get benchmark library
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Don't install benchmark" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Don't download dependencies" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Test in benchmark" FORCE)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
    unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
set_target_properties(benchmark PROPERTIES FOLDER "Extern")

add_executable(abm_benchmark_scaling_sim abm_benchmark_scaling.cpp)
target_link_libraries(abm_benchmark_scaling_sim PRIVATE abm benchmark::benchmark)

add_executable(abm_benchmark_strong_sim abm_benchmark_strong.cpp)
target_link_libraries(abm_benchmark_strong_sim PRIVATE memilio abm)
target_compile_options(abm_benchmark_strong_sim PRIVATE ${MEMILIO_CXX_FLAGS_ENABLE_WARNING_ERRORS})
