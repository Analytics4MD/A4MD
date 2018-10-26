set(_ingest_sources_
        ingest.cxx)
set(_ingest_headers_
        ingest.h)

include_directories(${PROJECT_SOURCE_DIR}/extern/dataspaces/include)

add_library(ingest ingest.cxx)



add_dependencies(ingest dataspaces)
target_link_libraries(ingest ${MPI_LIBRARIES})


install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
