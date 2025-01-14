cmake_minimum_required(VERSION 3.13)

project(memilio-simulations VERSION 1.0.0)

option(BUILD_2020_npis_sarscov2_wildtype_germany "Build 2024." ON)
option(BUILD_2021_vaccination_sarscov2_delta_germany "Build 2024." ON)
option(BUILD_abm "Build 2024." ON)
option(BUILD_abm_braunschweig "Build 2024." ON)
option(BUILD_munich_graph_sim "Build 2024." ON)
option(BUILD_2024_Ploetzke_Linear_Chain_Trick "Build 2024." ON)

# The Function BUILD_MEMILIO_VERSION is used to build a customized MEmilio version for each simulation.
# Arguments: 
#   NAME_MEMILIO_LIB: Name of the MEmilio library version.
#   GIT_TAG_MEMILIO: The git tag to be used (this can be a commit hash or a branch name).
#   USE_BUNDLED_THIRDPARTY: ON/OFF Specifies whether the options MEMILIO_USE_BUNDLED_... should be set to ON/OFF.
# TODO: Say that a tag is set to handle if this is the first time called
function(BUILD_MEMILIO_VERSION NAME_MEMILIO_LIB GIT_TAG_MEMILIO USE_BUNDLED_THIRDPARTY)
  set(MEMILIO_DOWNLOADED ON)
  # FetchContent to fetch the library in the correct version.
  include(FetchContent)
  
  FetchContent_Declare(
    ${NAME_MEMILIO_LIB}
    GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
    GIT_TAG ${GIT_TAG_MEMILIO}
  )
  
  FetchContent_GetProperties(${NAME_MEMILIO_LIB})
  
  if(NOT ${NAME_MEMILIO_LIB}_POPULATED)
    FetchContent_Populate(${NAME_MEMILIO_LIB})
  endif()

  # Disable some options for the build.
  set(MEMILIO_BUILD_TESTS OFF)
  set(MEMILIO_BUILD_EXAMPLES OFF)
  set(MEMILIO_BUILD_SIMULATIONS OFF)
  
  # Use bundled third-party dependencies.
  if(NOT USE_BUNDLED_THIRDPARTY)
    set(MEMILIO_USE_BUNDLED_SPDLOG OFF)
    set(MEMILIO_USE_BUNDLED_EIGEN OFF)
    set(MEMILIO_USE_BUNDLED_BOOST OFF)
    set(MEMILIO_USE_BUNDLED_JSONCPP OFF)
  endif()

  # Add the subdirectory for MEmilio build.
  add_subdirectory(${${NAME_MEMILIO_LIB}_SOURCE_DIR}/cpp ${${NAME_MEMILIO_LIB}_BINARY_DIR})
endfunction()


if(BUILD_2024_Ploetzke_Linear_Chain_Trick)
  # Read git tag.
  include(${CMAKE_SOURCE_DIR}/2024_Ploetzke_Linear_Chain_Trick/git_tag.cmake)
  BUILD_MEMILIO_VERSION("memilio_2024_lct" ${GIT_TAG_MEMILIO} ON)
  add_subdirectory(2024_Ploetzke_Linear_Chain_Trick)
endif()

if(BUILD_2020_npis_sarscov2_wildtype_germany)
  include(${CMAKE_SOURCE_DIR}/git_tag.cmake)
  if(MEMILIO_DOWNLOADED)
    BUILD_MEMILIO_VERSION("memilio_2020_ode" ${GIT_TAG_MEMILIO} OFF)
  else()
    BUILD_MEMILIO_VERSION("memilio_2020_ode" ${GIT_TAG_MEMILIO} ON)
  endif()
  if(MEMILIO_HAS_JSONCPP AND MEMILIO_HAS_HDF5)
    add_executable(2020_npis_wildtype 2020_npis_sarscov2_wildtype_germany.cpp)
    target_link_libraries(2020_npis_wildtype PRIVATE memilio ode_secir Boost::filesystem ${HDF5_C_LIBRARIES})
    target_compile_options(2020_npis_wildtype PRIVATE ${MEMILIO_CXX_FLAGS_ENABLE_WARNING_ERRORS})
  endif()
endif()

