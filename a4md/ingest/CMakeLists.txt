set(_ingest_sources_
        plumed_ingest.cxx
        md_ingest.cxx
        ingest.cxx 
        plumed_chunker.cxx
        chunker.cxx
        chunk_stager.cxx
        pycall.cxx)
set(_ingest_headers_
        plumed_ingest.h
        md_ingest.h
        ingest.h
        plumed_chunker.h
        chunker.h
        chunk_stager.h
        pycall.h
        )

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})

add_dependencies(ingest a4md_cmn dataspaces)
target_link_libraries(ingest a4md_cmn dataspaces dataspaces_cmn dataspaces_dart m ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
