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

# Add include directories
include_directories("${PROJECT_SOURCE_DIR}/a4md/common")
include_directories("${PROJECT_SOURCE_DIR}/a4md/dtl")
include_directories("${PROJECT_SOURCE_DIR}/a4md/ingest")
include_directories("${PROJECT_SOURCE_DIR}/a4md/retrieve")

# Add subdirectories
add_subdirectory(common)
add_subdirectory(dtl)
add_subdirectory(ingest)
add_subdirectory(retrieve)
add_subdirectory(extern)

# Define executables
# add_executable(ingester ingester.cxx)
# add_executable(retriever retriever.cxx)
#add_executable(mdgenerator mdgenerator.cxx)
add_executable(generator generator.cxx)
add_executable(analyzer analyzer.cxx)

add_subdirectory(${PROJECT_SOURCE_DIR}/a4md/extern/catch2)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
add_executable(tests-main tests-main.cpp test-ingest.cpp)
target_link_libraries(tests-main Catch2::Catch2)

# target_link_libraries(ingester ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
# target_link_libraries(retriever retrieve a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
#target_link_libraries(mdgenerator ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
target_link_libraries(generator a4md_cmn a4md_dtl a4md_ingest ${Boost_LIBRARIES} ${TAU_LIBRARIES})
target_link_libraries(analyzer a4md_cmn a4md_dtl a4md_retrieve ${Boost_LIBRARIES} ${TAU_LIBRARIES})

enable_testing()

## Install executables
# install (TARGETS ingester DESTINATION bin)
# install (TARGETS retriever DESTINATION bin)
#install (TARGETS mdgenerator DESTINATION bin)
install (TARGETS generator DESTINATION bin)
install (TARGETS analyzer DESTINATION bin)

include(CTest)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-main)
