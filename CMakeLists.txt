# project name
project (A4MD)
set(A4MD_VERSION 1.0)
cmake_minimum_required (VERSION 3.12)

option(dtl_dspaces "Build A4MD with Dataspaces data transport layer" ON)
option(dtl_dimes "Build A4MD with DIMES data transport layer" OFF)
option(dtl_decaf "Build A4MD with Decaf data transport layer" OFF)
option(build_examples "Build examples" ON)
option(build_tests "Build tests" OFF)
option(build_wrap "Build MPI wrapper library for Decaf MPMD model" OFF)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build libraries as .so")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_compile_options(-std=c++11)
message(STATUS "CFLAGS : ${CMAKE_C_FLAGS}")
message(STATUS "CXXFLAGS: ${CMAKE_CXX_FLAGS}")

# Set Cmake verbose
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Custom CMake scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#################################

## Update submodules
set (UPDATE_SUBMODULES 1 CACHE BOOL "Update git submodules for dependency projects")

## Add Git
find_package(Git)
if(GIT_FOUND)
  message("git found: ${GIT_EXECUTABLE}")
endif()
if ((EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git) AND GIT_EXECUTABLE)
    # only attempt to update submodules if this is a git repository and we have a git executable
    if (UPDATE_SUBMODULES)
      message(STATUS "Updating git submodules")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule --quiet sync
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
else()
    if (UPDATE_SUBMODULES)
    message(STATUS "Current source directory is " ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "UPDATE_SUBMODULES is set, but cmake cannot automatically update the submodules. "
                   "The source directory is not a git clone or the git executable is not available.")
    endif()
endif()

# Add MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    list(APPEND A4MD_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
    if (MPI_C_FOUND)
        message(STATUS "--==MPI C COMPILER FOUND==--" ${MPI_C_COMPILER})
        message(STATUS "MPI_C_COMPILE_FLAGS" ${MPI_C_COMPILE_FLAGS})
        message(STATUS "MPI_C_INCLUDE_PATH" ${MPI_C_INCLUDE_PATH})
        message(STATUS "MPI_C_LIBRARIES" ${MPI_C_LIBRARIES})
    else (MPI_C_FOUND)
        message(FATAL_ERROR "--==MPI C COMPILER NOT FOUND==--")
    endif (MPI_C_FOUND)
    if (MPI_CXX_FOUND)
        message(STATUS "--==MPI CXX COMPILER FOUND==--" ${MPI_CXX_COMPILER})
        message(STATUS "MPI_CXX_COMPILE_FLAGS" ${MPI_CXX_COMPILE_FLAGS})
        message(STATUS "MPI_CXX_INCLUDE_PATH" ${MPI_CXX_INCLUDE_PATH})
        message(STATUS "MPI_CXX_LIBRARIES" ${MPI_CXX_LIBRARIES})
    else (MPI_CXX_FOUND)
        message(FATAL_ERROR "--==MPI CXX COMPILER NOT FOUND==--")
    endif (MPI_CXX_FOUND)
    if (MPI_FORTRAN_FOUND)
        message(STATUS "--==MPI FORTRAN COMPILER FOUND==--" ${MPI_FORTRAN_COMPILER})
        message(STATUS "MPI_FORTRAN_COMPILE_FLAGS" ${MPI_Fortran_COMPILE_FLAGS})
        message(STATUS "MPI_FORTRAN_INCLUDE_PATH" ${MPI_Fortran_INCLUDE_PATH})
        message(STATUS "MPI_FORTRAN_LIBRARIES" ${MPI_Fortran_LIBRARIES})
    else (MPI_FORTRAN_FOUND)
        message(AUTHOR_WARNING "---===MPI FORTRAN COMPILER (typically mpif90) WAS NOT FOUND===---.
                Dataspaces may not build successfully because of this. Not sure what to do about this yet.
                If you are in a conda environment, switching to the system may work. Try it. Thats what worked for me.==--")
    endif (MPI_FORTRAN_FOUND)
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)
set (MPI_LIBRARIES ${MPI_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

# find Boost
SET (BOOST_MIN_VERSION "1.55.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS iostreams serialization)
if (NOT Boost_FOUND)
  message(FATAL_ERROR "Fatal error: Boost (version >= 1.55) required.")
else()
  message(STATUS "Setting up BOOST")
  message(STATUS " Boost Include Directories : ${Boost_INCLUDE_DIRS}")
  message(STATUS " Boost Libraries : ${Boost_LIBRARIES}")
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND A4MD_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif (NOT Boost_FOUND)

# Add Python
# This policy forces CMake to find Python based on location. In this case,
# use of `Python3_ROOT_DIR` should work as expected. If this wasn't set,
# CMake may grab a completely different Python than the one requested.
cmake_policy(SET CMP0094 NEW)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (NOT ${Python3_VERSION_MAJOR} EQUAL 3)
    message(FATAL_ERROR "FindPython3 somehow found a Python install that wasn't Python 3")
endif ()
if (${Python3_VERSION_MINOR} LESS_EQUAL 3)
    message(FATAL_ERROR "Found Python 3.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}, which has a version less than minimum allowed of 3.3")
endif ()
set(PYTHON_LIBRARIES Python3::Python)

# NOTE: Search for NumPy is commented out because NumPy is no longer used
#       in C++ files, and we do not need anything in CMake for a Python import of NumPy.
# find Numpy
# find_package(NumPy REQUIRED)
# include_directories(${NUMPY_INCLUDE_DIR})

# Pthread
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

if (dtl_dspaces OR dtl_dimes)
    find_package(dspaces CONFIG REQUIRED)
endif ()
if (dtl_dimes)
    add_definitions(-DDTL_DIMES=1)
endif ()

if (dtl_decaf)
    # Find Decaf
    find_package(Decaf REQUIRED)
    message(STATUS "Found external Decaf installed at: ${DECAF_PREFIX}")
    message(STATUS "DECAF_CXX_INCLUDE_DIR: ${DECAF_CXX_INCLUDE_DIR}")
    message(STATUS "DECAF_CXX_DATA_MODEL_LIBRARY: ${DECAF_CXX_DATA_MODEL_LIBRARY}")
    message(STATUS "DECAF_CXX_TRANSPORT_LIBRARY: ${DECAF_CXX_TRANSPORT_LIBRARY}")
    message(STATUS "DECAF_CXX_RUNTIME_LIBRARY: ${DECAF_CXX_RUNTIME_LIBRARY}")
    add_definitions(-DTRANSPORT_MPI)
    include_directories(${DECAF_CXX_INCLUDE_DIR})
    list(APPEND A4MD_INCLUDE_DIRS ${DECAF_CXX_INCLUDE_DIR})
    set(DECAF_LIBRARIES ${DECAF_CXX_TRANSPORT_LIBRARY} ${DECAF_CXX_DATA_MODEL_LIBRARY} ${DECAF_CXX_RUNTIME_LIBRARY})
    list(APPEND DTL_LIBRARIES ${DECAF_LIBRARIES})
endif ()

# Check TAU enabled
set(TAU_LIBRARIES "")
if(DEFINED ENV{TAU_LIBS})
    set(TAU_LIBRARIES "$ENV{TAU_LIBS}")
    message("TAU libraries " ${TAU_LIBRARIES})
    #add_definitions(-DTAU_PERF)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    message(STATUS "CXXFLAGS: ${CMAKE_CXX_FLAGS}")
endif()

# Check BUILT_IN_PERF
if(BUILT_IN_PERF)
    message("BUILT_IN_PERF is enabled")
    add_definitions(-DBUILT_IN_PERF)
endif()

# Check COUNT_LOST_FRAMES feature
if(COUNT_LOST_FRAMES)
    message("COUNT_LOST_FRAMES is enabled")
    add_definitions(-DCOUNT_LOST_FRAMES)
endif()

# Check DSPACES_DEBUG
if(DSPACES_DEBUG)
    message("DSPACES_DEBUG is enabled")
    add_definitions(-DDEBUG)
endif()

# Add include directories
include_directories("${PROJECT_SOURCE_DIR}/src/a4md/common")
include_directories("${PROJECT_SOURCE_DIR}/src/a4md/dtl")
include_directories("${PROJECT_SOURCE_DIR}/src/a4md/ingest")
include_directories("${PROJECT_SOURCE_DIR}/src/a4md/retrieve")

# Import pybind11
add_subdirectory(${PROJECT_SOURCE_DIR}/src/a4md/extern/pybind11)
# Build A4MD main sources
add_subdirectory(src/a4md)
if (build_examples)
    add_subdirectory(examples)
endif(build_examples)
if(build_tests)
    add_subdirectory(tests)
endif(build_tests)
if (build_wrap)
    add_subdirectory(wrap)
endif(build_wrap)
add_subdirectory(config)
