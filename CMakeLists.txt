cmake_minimum_required(VERSION 3.9)

project(
    A4MD
    VERSION 1.0
    LANGUAGES C CXX Fortran
)

set(A4MD_VERSION ${PROJECT_VERSION})

# A4MD uses C++11, so set the corresponding flag using CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Disable verbose Makefiles
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Enable and disable certain features of BLT
set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")
set(ENABLE_TESTS OFF CACHE BOOL "")
set(ENABLE_EXAMPLES OFF CACHE BOOL "")
set(ENABLE_GIT ON CACHE BOOL "")
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(ENABLE_MPI ON CACHE BOOL "")
set(ENABLE_GTEST OFF CACHE BOOL "")
set(ENABLE_GMOCK OFF CACHE BOOL "")
set(ENABLE_FRUIT OFF CACHE BOOL "")
set(ENABLE_FRUIT_MPI OFF CACHE BOOL "")
set(ENABLE_GBENCHMARK OFF CACHE BOOL "")

# Find BLT and include it
#
############################################################################
# This code is obtained from:                                              #
# https://llnl-blt.readthedocs.io/en/develop/tutorial/getting_started.html #
############################################################################
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    # NOTE: Change the path to point to the correct directory
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/a4md/extern/blt" CACHE PATH "")
    message(STATUS "Searching for BLT in ${BLT_SOURCE_DIR}")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT git submodule is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

option(ENABLE_DSPACES "Enable the DataSpaces Data Transport Layer" ON)
option(ENABLE_DIMES "Enable the DIMES Data Transport Layer" OFF)
option(ENABLE_DECAF "Enable the Decaf Data Transport Layer" OFF)
option(ENABLE_TAU "Enable TAU annotations" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_WRAP "Build MPI wrapper library for Decaf MPMD model" OFF)
option(UPDATE_SUBMODULES "Update git submodules for dependencies" ON)

# Update "real" options using old options if provided through the cache
if (DEFINED dtl_dspaces)
    set(ENABLE_DSPACES ${dtl_dspaces} CACHE BOOL "")
endif (DEFINED dtl_dspaces)
if (DEFINED dtl_dimes)
    set(ENABLE_DIMES ${dtl_dimes} CACHE BOOL "")
endif (DEFINED dtl_dimes)
if (DEFINED dtl_decaf)
    set(ENABLE_DECAF ${dtl_decaf} CACHE BOOL "")
endif (DEFINED dtl_decaf)
if (DEFINED build_examples)
    set(BUILD_EXAMPLES ${build_examples} CACHE BOOL "")
endif (DEFINED build_examples)
if (DEFINED build_tests)
    set(BUILD_TESTS ${build_tests} CACHE BOOL "")
endif (DEFINED build_tests)
if (DEFINED build_wrap)
    set(BUILD_WRAP ${build_wrap} CACHE BOOL "")
endif (DEFINED build_wrap)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR/cmake})

if (UPDATE_SUBMODULES)
    blt_is_git_repo(OUTPUT_STATE is_valid_repo)
    if (is_valid_repo)
        blt_git(
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            GIT_COMMAND "git submodule update --init --recursive"
            OUTPUT_VARIABLE git_submod_update_out
            RETURN_CODE git_submod_rc
        )
        if (NOT ${git_submod_rc} EQUAL 0)
            message(WARNING "UPDATE_SUBMODULES was enabled, but corresponding git command failed!")
        endif ()
    endif (is_valid_repo)
endif (UPDATE_SUBMODULES)

find_package(
    Boost 1.55.0
    REQUIRED
    COMPONENTS iostreams serialization
)
blt_import_library(
    NAME boost
    DEPENDS_ON Boost::boost Boost::iostreams Boost::serialization
)

if (${CMAKE_VERSION} VERSION_LESS "3.14.0")
    find_package(PythonInterp 3.6 REQURIED)
    set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)
    find_package(NumPy REQUIRED)
    blt_import_library(
        NAME python_dev
        INCLUDES ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR}
        LIBRARIES ${PYTHON_LIBRARIES}
        EXPORTABLE ON
    )
else ()
    find_package(
        Python3 3.6
        REQUIRED
        COMPONENTS Interpreter Development NumPy
    )
    blt_import_library(
        NAME python_dev
        INCLUDES ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS}
        LIBRARIES ${Python3_LIBRARIES}
        EXPORTABLE ON
    )
endif ()

find_package(Threads REQUIRED)

if (ENABLE_DSPACES)
    find_package(DataSpaces)
    if (DATASPACES_FOUND)
        message(STATUS "Found external DataSpaces installated at: ${DATASPACES_PREFIX}")
        message(STATUS "  * Version: ${DATASPACES_VERSION}")
        message(STATUS "  * Include Directories: ${DATASPACES_INCLUDE_DIR}")
        message(status "  * Libraries: ${DATASPACES_LIBRARIES}")
        blt_import_library(
            NAME dataspaces
            INCLUDES ${DATASPACES_INCLUDE_DIR}
            LIBRARIES ${DATASPACES_LIBRARIES}
            EXPORTABLE ON
        )
    endif (DATASPACES_FOUND)
endif (ENABLE_DSPACES)

if (ENABLE_DECAF)
    find_package(Decaf REQUIRED)
    message(STATUS "Found external Decaf installed at: ${DECAF_PREFIX}")
    message(STATUS "  * Include Dir: ${DECAF_CXX_INCLUDE_DIR}")
    message(STATUS "  * Data Model Library: ${DECAF_CXX_DATA_MODEL_LIBRARY}")
    message(STATUS "  * Transport Library: ${DECAF_CXX_DATA_TRANSPORT_LIBRARY}")
    message(STATUS "  * Runtime Library: ${DECAF_CXX_RUNTIME_LIBRARY}")
    add_definitions(-DTRANSPORT_MPI)
    blt_import_library(
        NAME decaf
        INCLUDES ${DECAF_CXX_INCLUDE_DIR}
        LIBRARIES ${DECAF_CXX_DATA_MODEL_LIBRARY}
                  ${DECAF_CXX_DATA_TRANSPORT_LIBRARY} 
                  ${DECAF_CXX_RUNTIME_LIBRARY}
        EXPORTABLE ON
    )
endif (ENABLE_DECAF)

if (ENABLE_TAU)
    if (DEFINED ENV{TAU_LIBS})
        blt_import_library(
            NAME tau
            LIBRARIES "$ENV{TAU_LIBS}"
            COMPILE_FLAGS "-g"
            EXPORTABLE ON
        )
    else (DEFINED ENV{TAU_LIBS})
        find_package(TAU REQUIRED)
        blt_import_library(
            NAME tau
            LIBRARIES ${TAU_LIBRARIES}
            INCLUDES ${TAU_INCLUDE_DIR}
            COMPILE_FLAGS "-g"
            EXPORTABLE ON
        )
    endif (DEFINED ENV{TAU_LIBS})
endif (ENABLE_TAU)

if (BUILT_IN_PERF)
    message(STATUS "BUILT_IN_PERF enabled")
    add_definitions(-DBUILT_IN_PERF)
endif (BUILT_IN_PERF)

if (COUNT_LOST_FRAMES)
    message(STATUS "COUNT_LOST_FRAMES enabled")
    add_definitions(-DCOUNT_LOST_FRAMES)
endif (COUNT_LOST_FRAMES)

if (DSPACES_DEBUG)
    message(STATUS "DSPACES_DEBUG enabled")
    add_definitions(-DDEBUG)
endif (DSPACES_DEBUG)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/a4md/extern/pybind11)

add_subdirectory(src/a4md)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif (BUILD_EXAMPLES)
if (BUILD_TESTS)
    add_subdirectory(tests)
endif (BUILD_TESTS)
if (BUILD_WRAP)
    add_subdirectory(wrap)
endif (BUILD_WRAP)
add_subdirectory(config)