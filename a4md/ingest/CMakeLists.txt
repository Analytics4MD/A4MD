set(_ingest_sources_
        ingest.cxx
        chunker.cxx
        pycall.cxx
        chunk_reader.cxx
        ims_reader.cxx)
set(_ingest_headers_
        ingest.h
        pycall.h
        chunker.h
        chunk_reader.h
        ims_reader.h
        common.h
        exceptions.h)

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})



add_dependencies(ingest dataspaces)
target_link_libraries(ingest dataspaces dataspaces_cmn dataspaces_dart m ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
