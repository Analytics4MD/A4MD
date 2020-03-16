include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
set(cereal_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cereal)
ExternalProject_Add(
    cereal
    PREFIX ${cereal_PREFIX}
    URL "${CMAKE_CURRENT_SOURCE_DIR}/serialization_libraries/cereal/cereal-1.2.2.tar.gz"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND mkdir -p ${cereal_PREFIX}/include/ && cp -r ${cereal_PREFIX}/src/cereal/include/cereal ${cereal_PREFIX}/include/
)
include_directories(${cereal_PREFIX}/include)

set(yas_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/yas)
ExternalProject_Add(
    yas
    PREFIX ${yas_PREFIX}
    URL "${CMAKE_CURRENT_SOURCE_DIR}/serialization_libraries/yas/yas-7.0.2.tar.gz"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND mkdir -p ${yas_PREFIX}/include/ && cp -r ${yas_PREFIX}/src/yas/include/yas ${yas_PREFIX}/include/
)
include_directories(${yas_PREFIX}/include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(test_serialization test_serialization.cxx)
add_dependencies(test_serialization a4md_cmn cereal yas)
target_link_libraries(test_serialization a4md_cmn ${MPI_LIBRARIES} ${Boost_LIBRARIES} pthread)

