set(_ingest_sources_
<<<<<<< HEAD
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
=======
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
>>>>>>> feat/ingest_library
        common.h
        exceptions.h)

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ${_ingest_sources_})



add_dependencies(ingest dataspaces)
target_link_libraries(ingest dataspaces dataspaces_cmn dataspaces_dart m ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
