cmake_minimum_required (VERSION 2.8.12)
add_compile_options(-std=c++11)
project (a4md)
set(CMAKE_VERBOSE_MAKEFILE ON)
message("CFLAGS : ${CMAKE_C_FLAGS}")
message("CXXFLAGS: ${CMAKE_CXX_FLAGS}")
#################################
## update submodules
set (UPDATE_SUBMODULES 1 CACHE BOOL "Update git submodules for dependency projects")
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

# Custom CMake scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(nersc FALSE)
if(DEFINED ENV{NERSC_HOST})
    message("The code is built at NERSC/" $ENV{NERSC_HOST})
    set(nersc TRUE)
    message(STATUS "NERSC build is enabled")
endif()

set(TAU_LIBRARIES "")
if(DEFINED ENV{TAU_LIBS})
    set(TAU_LIBRARIES "$ENV{TAU_LIBS}")
    message("TAU libraries " ${TAU_LIBRARIES})
    add_definitions(-DTAU_PERF)
endif()

if(BUILT_IN_PERF)
    message("BUILT_IN_PERF is enabled")
    add_definitions(-DBUILT_IN_PERF)
endif()

if(COUNT_LOST_FRAMES)
    message("COUNT_LOST_FRAMES is enabled")
    add_definitions(-DCOUNT_LOST_FRAMES)
endif()
add_subdirectory(a4md)
