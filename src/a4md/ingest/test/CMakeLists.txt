add_executable(tests-ingest tests-ingest.cxx tests-ingest-ds.cxx)
target_link_libraries(tests-ingest Catch2::Catch2 a4md_ingest)
include(CTest)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-ingest)

add_custom_target(
	CopyTestInputs ALL
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ingest.py ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test.pdb ${CMAKE_CURRENT_BINARY_DIR}
	)
