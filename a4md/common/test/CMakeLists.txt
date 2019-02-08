add_executable(tests-cmn test-main.cxx test-ds.cxx)
target_link_libraries(tests-cmn Catch2::Catch2 a4md_cmn)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-cmn)

