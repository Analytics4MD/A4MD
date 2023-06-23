# Add subdirectories
add_subdirectory(common)
add_subdirectory(dtl)
add_subdirectory(ingest)
add_subdirectory(retrieve)
add_subdirectory(extern)

if (BUILD_EXAMPLES OR BUILD_TESTS)
    # Define executables
    # add_executable(ingester ingester.cxx)
    # add_executable(retriever retriever.cxx)
    #add_executable(mdgenerator mdgenerator.cxx)
	blt_add_exectuable(
		NAME generator
		SOURCES generator.cxx
		DEPENDS_ON a4md_cmn a4md_dtl a4md_ingest boost
	)
	blt_add_exectuable(
		NAME analyzer
		SOURCES analyzer.cxx
		DEPENDS_ON a4md_cmn a4md_dtl a4md_retrieve boost
	)
	blt_add_exectuable(
		NAME test_producer
		SOURCES test_producer.cxx
		DEPENDS_ON a4md_cmn a4md_dtl a4md_ingest boost
	)
	blt_add_exectuable(
		NAME test_consumer
		SOURCES test_consumer.cxx
		DEPENDS_ON a4md_cmn a4md_dtl a4md_retrieve boost
	)
    if (ENABLE_TAU)
		blt_patch_target(
			NAME generator
			DEPENDS_ON tau
		)
		blt_patch_target(
			NAME analyzer
			DEPENDS_ON tau
		)
		blt_patch_target(
			NAME test_producer
			DEPENDS_ON tau
		)
		blt_patch_target(
			NAME test_consumer
			DEPENDS_ON tau
		)
	endif (ENABLE_TAU)

    # target_link_libraries(ingester ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
    # target_link_libraries(retriever retrieve a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})
    #target_link_libraries(mdgenerator ingest a4md_cmn ${Boost_LIBRARIES} ${TAU_LIBRARIES})

    ## Install executables
    # install (TARGETS ingester DESTINATION bin)
    # install (TARGETS retriever DESTINATION bin)
    #install (TARGETS mdgenerator DESTINATION bin)
	install (TARGETS generator DESTINATION bin)
	install (TARGETS analyzer DESTINATION bin)
	install (TARGETS test_producer DESTINATION bin)
	install (TARGETS test_consumer DESTINATION bin)

	if (dtl_decaf)
		blt_add_exectuable(
			NAME generator_decaf
			SOURCES generator_decaf.cxx
			DEPENDS_ON a4md_cmn a4md_dtl a4md_ingest boost
		)
		blt_add_exectuable(
			NAME analyzer_decaf
			SOURCES analyzer_decaf.cxx
			DEPENDS_ON a4md_cmn a4md_dtl a4md_retrieve boost
		)
	    if (ENABLE_TAU)
			blt_patch_target{
				NAME generator_decaf
				DEPENDS_ON tau
			}
			blt_patch_target{
				NAME analyzer_decaf
				DEPENDS_ON tau
			}
		endif (ENABLE_TAU)
		install (TARGETS generator_decaf DESTINATION bin)
		install (TARGETS analyzer_decaf DESTINATION bin)
	endif()

	add_subdirectory(${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2)
	set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/a4md/extern/catch2/contrib" ${CMAKE_MODULE_PATH})
	blt_add_exectuable(
		NAME tests-main
		SOURCES tests-main.cpp test-ingest.cpp
		DEPENDS_ON Catch2::Catch2
	)
	enable_testing()
	include(CTest)
	include(ParseAndAddCatchTests)
	ParseAndAddCatchTests(tests-main)
endif (BUILD_EXAMPLES OR BUILD_TESTS)
	