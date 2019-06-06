set(_ingest_sources_
        ingester.cxx
        md_generator.cxx
        plumed_chunker.cxx
        chunk_stager.cxx
        md_stager.cxx
		pdb_chunker.cxx
		)
set(_ingest_headers_
        ingester.h
        md_generator.h
        plumed_chunker.h
        chunk_stager.h
        md_stager.h
		pdb_chunker.h
        )

#include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})
set_target_properties(ingest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_dependencies(ingest a4md_cmn)
target_link_libraries(ingest a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES})

add_subdirectory(test)
install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
