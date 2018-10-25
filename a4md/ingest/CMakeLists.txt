set(_ingest_sources_
        ingest.cxx)
set(_ingest_headers_
        ingest.h)

add_library(ingest ingest.cxx)

install (TARGETS ingest DESTINATION lib)
install (FILES ${_ingest_headers_} DESTINATION include)
