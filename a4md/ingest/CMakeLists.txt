set(_ingest_sources_
        plumed_ingest.cxx
        ingest.cxx 
        chunker.cxx
        chunk_stager.cxx
        chunk_writer.cxx
        chunk_reader.cxx)
set(_ingest_headers_
        plumed_ingest.h
        ingest.h
        chunker.h
        chunk_stager.h
        chunk_writer.h
        chunk_reader.h
        ims_reader.h
        ims_writer.h
        common.h
        exceptions.h)

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})



add_dependencies(ingest dataspaces)
target_link_libraries(ingest dataspaces dataspaces_cmn dataspaces_dart m rdmacm ibverbs ${MPI_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
