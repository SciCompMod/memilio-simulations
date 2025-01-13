cmake_minimum_required(VERSION 3.13)

project(memilio-simulations VERSION 1.0.0)

option(MEMILIO_2024_PLOETZKE "Build 2024." ON)

function(download_memilio_version NAME_MEM GIT_TAG_MEMILIO USE_BUNDLED)


# If git tag is not set, throw error.
if(NOT DEFINED GIT_TAG_MEMILIO)
  message(FATAL_ERROR "GIT_TAG_MEMILIO is not defined. Please make sure the git_tag.cmake file is correct.")
endif()

# FetchContent to fetch the library with git tag defined.
include(FetchContent)
FetchContent_Declare(
  memilio
  GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
  GIT_TAG ${GIT_TAG_MEMILIO}
)
FetchContent_GetProperties(memilio)

if(NOT memilio_POPULATED)
  FetchContent_Populate(memilio)
endif()

# Disable some options.
set(MEMILIO_BUILD_TESTS ON)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)
# Add CMake-Project and build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

endfunction(download_memilio_version git_tag_mem)

# Define function for this with name (memiliov1?) git tag and possibly ON/OFF for USE_BUNDLED.

if(MEMILIO_2024_PLOETZKE)
  # Read git tag.
  include(${CMAKE_SOURCE_DIR}/2024_Ploetzke_Linear_Chain_Trick/git_tag.cmake)
    add_subdirectory(2024_Ploetzke_Linear_Chain_Trick)
endif()

