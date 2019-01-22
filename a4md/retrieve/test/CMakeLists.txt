add_executable(tests-retrieve test-main.cxx)
target_link_libraries(tests-retrieve Catch2::Catch2 retrieve)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-retrieve)

