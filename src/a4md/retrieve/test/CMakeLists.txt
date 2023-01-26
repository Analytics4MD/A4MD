add_executable(tests-retrieve test-main.cxx)
target_link_libraries(tests-retrieve Catch2::Catch2 a4md_retrieve)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-retrieve)

add_custom_target(
	CopyRetrieveTestInputs ALL
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test_analysis.py ${CMAKE_CURRENT_BINARY_DIR}
	)
