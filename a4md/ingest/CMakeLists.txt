set(_a4md_ingest_sources_
        ingester.cxx
        md_generator.cxx
        # chunk_stager.cxx
        # md_stager.cxx
		pdb_chunker.cxx
        cv_buffer.cxx
		)
set(_a4md_ingest_headers_
        ingester.h
        md_generator.h
        # chunk_stager.h
        # md_stager.h
		pdb_chunker.h
        cv_buffer.h
        )

#include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(a4md_ingest ${_a4md_ingest_sources_})
set_target_properties(a4md_ingest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_dependencies(a4md_ingest a4md_cmn a4md_dtl)
target_link_libraries(a4md_ingest a4md_cmn a4md_dtl ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES})

add_subdirectory(test)
install (TARGETS a4md_ingest DESTINATION lib)
install (FILES ${_a4md_ingest_headers_} DESTINATION include)
