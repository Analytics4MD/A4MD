# adding python dependency 
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# adding MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

# adding Dataspaces library
message(STATUS "Adding Dataspaces library to project")
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
ExternalProject_Add(project_dataspaces
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/a4md/extern/dataspaces
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure CC=mpicc FC=mpif90 --prefix=${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces
  BUILD_COMMAND ${MAKE}
)
add_library(dataspaces STATIC IMPORTED)
add_dependencies(dataspaces project_dataspaces)
set_property(TARGET dataspaces PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdspaces.a)

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
target_link_libraries(ingester ingest)
target_link_libraries(retriever retrieve)

install (TARGETS ingester DESTINATION bin)
install (TARGETS retriever DESTINATION bin)

