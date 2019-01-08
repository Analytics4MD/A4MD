add_executable(tests-cmn test-main.cxx)
target_link_libraries(tests-cmn Catch2::Catch2 a4md_cmn)
include(CTest)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests-cmn)

