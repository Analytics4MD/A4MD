# find python interpreter
find_package(PythonInterp 3.6 REQUIRED)

# find python libs
find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# find Numpy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIR})

# adding Dataspaces library
message(STATUS "Adding Dataspaces library to project")
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set(DS_CONF_ENABLES "--enable-dart-tcp")
if (nersc)
    set(DS_CONF_ENABLES "--enable-drc;--enable-dimes;--enable-shmem")
else (nersc)
    # ToDo: Fix MPI_FORTRAN not found to unhardcode this
    set(MPI_FORTRAN_COMPILER "mpif90")
endif (nersc)

ExternalProject_Add(project_dataspaces
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/a4md/extern/dataspaces
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure CC=${MPI_C_COMPILER} CXX=${MPI_CXX_COMPILER} FC=${MPI_FORTRAN_COMPILER} ${DS_CONF_ENABLES} --prefix=${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces
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
include_directories("${PROJECT_SOURCE_DIR}/a4md/common")
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
add_executable(analyzer analyzer.cxx)
#add_executable(mdgenerator mdgenerator.cxx)
add_executable(generator generator.cxx)

add_subdirectory(${PROJECT_SOURCE_DIR}/a4md/extern/catch2)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})

add_executable(tests-main tests-main.cpp test-ingest.cpp)
target_link_libraries(tests-main Catch2::Catch2)
target_link_libraries(ingester ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
target_link_libraries(retriever retrieve a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
target_link_libraries(analyzer retrieve a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
#target_link_libraries(mdgenerator ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
target_link_libraries(generator ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})

enable_testing()
install (TARGETS ingester DESTINATION bin)
install (TARGETS retriever DESTINATION bin)
install (TARGETS analyzer DESTINATION bin)
#install (TARGETS mdgenerator DESTINATION bin)
install (TARGETS generator DESTINATION bin)
include(CTest)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-main)
