cmake_minimum_required(VERSION 3.18)

project(memilio-simulations-2026_Zunker_et_al_High_performance VERSION 1.0.0 LANGUAGES C CXX)

option(HPC_METAPOP_CUDA "Build the CUDA benchmarks (requires the CUDA toolkit)." OFF)
option(HPC_METAPOP_NATIVE "Optimize for the build host (-march=native), as in the paper." ON)
option(HPC_METAPOP_ROOFLINE "Build with the roofline profiling hooks (only for roofline measurements)." OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(${CMAKE_SOURCE_DIR}/git_tag.cmake)

if(NOT DEFINED GIT_TAG_MEMILIO)
  message(FATAL_ERROR "GIT_TAG_MEMILIO is not defined. Please make sure the git_tag.cmake file is correct.")
endif()

include(FetchContent)

FetchContent_Declare(
  memilio
  GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
  GIT_TAG ${GIT_TAG_MEMILIO}
  SOURCE_SUBDIR cpp
)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF CACHE BOOL "")
set(MEMILIO_BUILD_EXAMPLES OFF CACHE BOOL "")
set(MEMILIO_BUILD_SIMULATIONS OFF CACHE BOOL "")
set(MEMILIO_ENABLE_OPENMP ON CACHE BOOL "")
set(MEMILIO_ENABLE_HDF5 OFF CACHE BOOL "")

FetchContent_MakeAvailable(memilio)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.6.1)
FetchContent_MakeAvailable(benchmark)

find_package(OpenMP REQUIRED)

set(native_flag $<$<AND:$<BOOL:${HPC_METAPOP_NATIVE}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-march=native>)

add_executable(ode_seir_mobility_outcomes ode_seir_mobility_outcomes.cpp)
target_link_libraries(ode_seir_mobility_outcomes PRIVATE memilio ode_seir_metapop)

add_executable(ode_seir_benchmark_stage_aligned ode_seir_benchmark_stage_aligned.cpp)
target_link_libraries(ode_seir_benchmark_stage_aligned PRIVATE memilio benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(ode_seir_benchmark_stage_aligned PRIVATE ${native_flag})

add_executable(ode_seir_metapop_benchmark_cpu ode_seir_metapop_benchmark_cpu.cpp)
target_link_libraries(ode_seir_metapop_benchmark_cpu PRIVATE Eigen3::Eigen benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_definitions(ode_seir_metapop_benchmark_cpu PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(ode_seir_metapop_benchmark_cpu PRIVATE ${native_flag})

set(benchmark_targets ode_seir_benchmark_stage_aligned ode_seir_metapop_benchmark_cpu)

# GPU application
if(HPC_METAPOP_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(ode_seir_benchmark_gpu ode_seir_benchmark_gpu.cpp ode_seir_benchmark_gpu_kernels.cu)
  target_link_libraries(ode_seir_benchmark_gpu PRIVATE benchmark::benchmark CUDA::cudart)

  add_executable(ode_seir_metapop_benchmark_gpu ode_seir_metapop_benchmark_gpu.cu)
  target_link_libraries(ode_seir_metapop_benchmark_gpu PRIVATE Eigen3::Eigen benchmark::benchmark CUDA::cudart CUDA::cublas)
  target_compile_definitions(ode_seir_metapop_benchmark_gpu PRIVATE EIGEN_DONT_PARALLELIZE EIGEN_NO_CUDA)

  if(HPC_METAPOP_NATIVE)
    target_compile_options(ode_seir_metapop_benchmark_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
  endif()

  set_target_properties(ode_seir_benchmark_gpu ode_seir_metapop_benchmark_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
  list(APPEND benchmark_targets ode_seir_benchmark_gpu ode_seir_metapop_benchmark_gpu)
endif()

if(HPC_METAPOP_ROOFLINE)
  foreach(target ${benchmark_targets})
    target_compile_definitions(${target} PRIVATE MEMILIO_BENCHMARK_ROOFLINE)
  endforeach()
endif()
