add_executable(tests-cmn test-chunk.cxx test-runner.cxx test-chunk-serializer.cxx)
target_link_libraries(tests-cmn Catch2::Catch2 a4md_cmn)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-cmn)

add_custom_target(
	CopyCommonTestInputs ALL
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/direct.py ${CMAKE_CURRENT_BINARY_DIR}
	)

