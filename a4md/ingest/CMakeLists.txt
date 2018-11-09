set(_ingest_sources_
        ingest.cxx
        chunk_stager.cxx
        chunk_reader.cxx
        chunk_writer.cxx
        chunker.cxx
        ims_reader.cxx
        pycall.cxx)
set(_ingest_headers_
        ingest.h
        chunk_stager.h
        chunk_reader.h
        chunk_writer.h
        pycall.h
        chunker.h
        ims_reader.h
        common.h
        exceptions.h)

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})



add_dependencies(ingest dataspaces)
target_link_libraries(ingest dataspaces dataspaces_cmn dataspaces_dart m ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
