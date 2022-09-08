set(_a4md_ingest_sources_
        src/ingester.cxx
        src/md_generator.cxx
        # src/chunk_stager.cxx
        # src/md_stager.cxx
	src/pdb_chunker.cxx
        src/cv_buffer.cxx
		)
set(_a4md_ingest_headers_
        include/ingester.h
        include/md_generator.h
        # include/chunk_stager.h
        # include/md_stager.h
	include/pdb_chunker.h
        include/cv_buffer.h
        )

#include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(a4md_ingest ${_a4md_ingest_sources_})
target_include_directories(a4md_ingest PUBLIC include/)
set_target_properties(a4md_ingest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_dependencies(a4md_ingest a4md_cmn a4md_dtl)
target_link_libraries(a4md_ingest a4md_cmn a4md_dtl ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES})
get_target_property(A4MD_INGEST_LIBRARIES a4md_ingest LINK_LIBRARIES)
message(STATUS "A4MD_INGEST_LIBRARIES = ${A4MD_INGEST_LIBRARIES}")

add_subdirectory(test)
install (TARGETS a4md_ingest DESTINATION lib)
install (FILES ${_a4md_ingest_headers_} DESTINATION include)
