set(_ingest_sources_
        plumed_ingest.cxx
        md_ingest.cxx
        ingest.cxx 
        plumed_chunker.cxx
        chunk_stager.cxx
        pycall.cxx)
set(_ingest_headers_
        plumed_ingest.h
        md_ingest.h
        ingest.h
        plumed_chunker.h
        chunk_stager.h
        pycall.h
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
target_link_libraries(ingest a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
