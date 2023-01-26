add_executable(tests-dtl test-ds.cxx)
target_link_libraries(tests-dtl Catch2::Catch2 a4md_dtl)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-dtl)

