# find python interpreter
find_package(PythonInterp 3.6 REQUIRED)

# find python libs
find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# find Numpy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIR})

# find Boost
SET (BOOST_MIN_VERSION "1.55.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS iostreams serialization)
if (NOT Boost_FOUND)
  message(FATAL_ERROR "Fatal error: Boost (version >= 1.55) required.")
else()
  message(STATUS "Setting up BOOST")
  message(STATUS " Includes - ${Boost_INCLUDE_DIRS}")
  message(STATUS " Library  - ${Boost_LIBRARY_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif (NOT Boost_FOUND)
# adding MPI
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
    else (MPI_CXX_FOUND)
        message(FATAL_ERROR "--==MPI CXX COMPILER NOT FOUND==--")
    endif (MPI_CXX_FOUND)
    if (MPI_FORTRAN_FOUND)
        message(STATUS "--==MPI FORTRAN COMPILER FOUND==--" ${MPI_FORTRAN_COMPILER})
    else (MPI_FORTRAN_FOUND)
        message(STATUS "MPI_FORTRSN_COMPILE_FLAGS" ${MPI_Fortran_COMPILE_FLAGS})
        message(STATUS "MPI_FORTRAN_INCLUDE_PATH" ${MPI_Fortran_INCLUDE_PATH})
        message(STATUS "MPI_FORTRAN_LIBRARIES" ${MPI_Fortran_INCLUDE_PATH})

        message(AUTHOR_WARNING "---===MPI FORTRAN COMPILER (typically mpif90) WAS NOT FOUND===---.
                Dataspaces may not build successfully because of this. Not sure what to do about this yet.
                If you are in a conda environment, switching to the system may work. Try it. Thats what worked for me.==--")
    endif (MPI_FORTRAN_FOUND)
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

# adding Dataspaces library
message(STATUS "Adding Dataspaces library to project")
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

ExternalProject_Add(project_dataspaces
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/a4md/extern/dataspaces
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure CC=mpicc FC=mpif90 CFLAGS=-fPIC --enable-dart-tcp LIBS=-lm --prefix=${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces
  BUILD_COMMAND ${MAKE}
)
add_library(dataspaces STATIC IMPORTED)
add_dependencies(dataspaces project_dataspaces)
set_property(TARGET dataspaces PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdspaces.a)
set_target_properties(dataspaces PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)


add_library(dataspaces_cmn STATIC IMPORTED)
add_dependencies(dataspaces_cmn project_dataspaces)
set_property(TARGET dataspaces_cmn PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdscommon.a)

add_library(dataspaces_dart STATIC IMPORTED)
add_dependencies(dataspaces_dart project_dataspaces)
set_property(TARGET dataspaces_dart PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdart.a)



SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#set_property(TARGET dataspaces PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include)

#message("dataspaces include directory " ${CMAKE_CURRENT_BINARY_DIR}/dataspaces/include)
#include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

# add libraries
include_directories("${PROJECT_SOURCE_DIR}/a4md/ingest")
include_directories("${PROJECT_SOURCE_DIR}/a4md/retrieve")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/bin/" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING PATTERN "dataspaces_server")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include/" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        USE_SOURCE_PERMISSIONS)

add_subdirectory(common)
add_subdirectory(ingest)
add_subdirectory(retrieve)
add_subdirectory(extern)

# add project level sources
set(_a4md_sources_
        main.cxx)


# add executables
add_executable(a4md ${_a4md_sources_})
add_executable(ingester ingester.cxx)
add_executable(retriever retriever.cxx)
target_link_libraries(ingester ingest a4md_cmn ${Boost_LIBRARIES})
target_link_libraries(retriever retrieve a4md_cmn ${Boost_LIBRARIES})

install (TARGETS ingester DESTINATION bin)
install (TARGETS retriever DESTINATION bin)
