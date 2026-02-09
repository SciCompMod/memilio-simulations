cmake_minimum_required(VERSION 3.13)

project(memilio-simulations-2026-Bicker-et-al-Memilio-paper-Fig6_a VERSION 1.0.0)

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
set(MEMILIO_BUILD_EXAMPLES ON)
set(MEMILIO_BUILD_SIMULATIONS OFF)
set(MEMILIO_ENABLE_OPENMP ON)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

add_executable(ode_seir_metapop_benchmark ode_seir_metapop_benchmark.cpp)
target_link_libraries(ode_seir_metapop_benchmark PRIVATE memilio ode_seir_metapop)
target_compile_options(ode_seir_metapop_benchmark PRIVATE ${MEMILIO_CXX_FLAGS_ENABLE_WARNING_ERRORS})
