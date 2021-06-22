add_executable(tests-cmn test-chunk.cxx test-runner.cxx test-chunk-serializer.cxx test-py-executor.cxx)
target_link_libraries(tests-cmn Catch2::Catch2 a4md_cmn)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-cmn)

add_custom_target(
	CopyCommonTestInputs ALL
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/direct.py ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/tests.py ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test.pdb ${CMAKE_CURRENT_BINARY_DIR}
	)

