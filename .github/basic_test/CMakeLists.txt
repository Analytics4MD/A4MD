cmake_minimum_required(VERSION 3.12)
project(a4md_basic_test_ci)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)

cmake_policy(SET CMP0094 NEW)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (NOT ${Python3_VERSION_MAJOR} EQUAL 3)
    message(FATAL_ERROR "FindPython3 somehow found a Python install that wasn't Python 3")
endif ()
if (${Python3_VERSION_MINOR} LESS_EQUAL 3)
    message(FATAL_ERROR "Found Python 3.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}, which has a version less than minimum allowed of 3.3")
endif ()

find_package(Boost 1.59.0 REQUIRED COMPONENTS iostreams serialization)

find_package(A4MD REQUIRED)

add_executable(producer producer.cxx)
add_executable(consumer consumer.cxx)

target_link_libraries(producer MPI::MPI_CXX Python3::Python Boost::boost Boost::iostreams Boost::serialization ${A4MD_LIBRARIES})
target_link_libraries(consumer MPI::MPI_CXX Python3::Python Boost::boost Boost::iostreams Boost::serialization ${A4MD_LIBRARIES})
target_include_directories(producer PRIVATE ${A4MD_INCLUDE_DIRS})
target_include_directories(consumer PRIVATE ${A4MD_INCLUDE_DIRS})

install(TARGETS producer DESTINATION bin)
install(TARGETS consumer DESTINATION bin)
install(FILES flux_prod_cons.sh
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_WRITE GROUP_EXECUTE
                    WORLD_READ WORLD_WRITE WORLD_EXECUTE
)
install(FILES local_prod_cons.sh
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE
        WORLD_READ WORLD_WRITE WORLD_EXECUTE
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/script/"
        DESTINATION bin
        USE_SOURCE_PERMISSIONS
)
