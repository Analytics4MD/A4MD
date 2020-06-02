# project name
project (a4md)

cmake_minimum_required (VERSION 3.5)
add_compile_options(-std=c++11)
message(STATUS "CFLAGS : ${CMAKE_C_FLAGS}")
message(STATUS "CXXFLAGS: ${CMAKE_CXX_FLAGS}")

# Set Cmake verbose
set(CMAKE_VERBOSE_MAKEFILE OFF)

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
    if (MPI_C_FOUND)
        message(STATUS "--==MPI C COMPILER FOUND==--" ${MPI_C_COMPILER})
        message(STATUS "MPI_C_COMPILE_FLAGS" ${MPI_C_COMPILE_FLAGS})
        message(STATUS "MPI_C_INCLUDE_PATH" ${MPI_C_INCLUDE_PATH})
        message(STATUS "MPI_C_LIBRARIES" ${MPI_C_INCLUDE_PATH})
    else (MPI_C_FOUND)
        message(FATAL_ERROR "--==MPI C COMPILER NOT FOUND==--")
    endif (MPI_C_FOUND)
    if (MPI_CXX_FOUND)
        message(STATUS "--==MPI CXX COMPILER FOUND==--" ${MPI_CXX_COMPILER})
        message(STATUS "MPI_CXX_COMPILE_FLAGS" ${MPI_CXX_COMPILE_FLAGS})
        message(STATUS "MPI_CXX_INCLUDE_PATH" ${MPI_CXX_INCLUDE_PATH})
        message(STATUS "MPI_CXX_LIBRARIES" ${MPI_CXX_INCLUDE_PATH})
    else (MPI_CXX_FOUND)
        message(FATAL_ERROR "--==MPI CXX COMPILER NOT FOUND==--")
    endif (MPI_CXX_FOUND)
    if (MPI_FORTRAN_FOUND)
        message(STATUS "--==MPI FORTRAN COMPILER FOUND==--" ${MPI_FORTRAN_COMPILER})
        message(STATUS "MPI_FORTRAN_COMPILE_FLAGS" ${MPI_Fortran_COMPILE_FLAGS})
        message(STATUS "MPI_FORTRAN_INCLUDE_PATH" ${MPI_Fortran_INCLUDE_PATH})
        message(STATUS "MPI_FORTRAN_LIBRARIES" ${MPI_Fortran_INCLUDE_PATH})
    else (MPI_FORTRAN_FOUND)
        message(AUTHOR_WARNING "---===MPI FORTRAN COMPILER (typically mpif90) WAS NOT FOUND===---.
                Dataspaces may not build successfully because of this. Not sure what to do about this yet.
                If you are in a conda environment, switching to the system may work. Try it. Thats what worked for me.==--")
    endif (MPI_FORTRAN_FOUND)
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

# find Boost
SET (BOOST_MIN_VERSION "1.55.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS iostreams serialization)
if (NOT Boost_FOUND)
  message(FATAL_ERROR "Fatal error: Boost (version >= 1.55) required.")
else()
  message(STATUS "Setting up BOOST")
  message(STATUS " Boost Include Directories : ${Boost_INCLUDE_DIRS}")
  message(STATUS " Boost Libraries : ${Boost_LIBRARY_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif (NOT Boost_FOUND)

# Custom CMake scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# FindPythonInterp will search for an interpreter with 3.3 <= version <= 3.7
set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 3.4 3.3)
# find python interpreter (minimum version 3.3)
find_package(PythonInterp 3.3 REQUIRED)
# Reset Python_ADDITIONAL_VERSIONS so that FindPythonLibs only searches for the
# same version as the Interpreter.
unset(Python_ADDITIONAL_VERSIONS)
set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
# find python libs (minimum version matches the interpreter's version)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
# find Numpy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")



# Check TAU enabled
set(TAU_LIBRARIES "")
if(DEFINED ENV{TAU_LIBS})
    set(TAU_LIBRARIES "$ENV{TAU_LIBS}")
    message("TAU libraries " ${TAU_LIBRARIES})
    add_definitions(-DTAU_PERF)
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
include_directories("${PROJECT_SOURCE_DIR}/a4md/common")
include_directories("${PROJECT_SOURCE_DIR}/a4md/dtl")
include_directories("${PROJECT_SOURCE_DIR}/a4md/ingest")
include_directories("${PROJECT_SOURCE_DIR}/a4md/retrieve")

add_subdirectory(a4md)
add_subdirectory(examples)
add_subdirectory(tests)
