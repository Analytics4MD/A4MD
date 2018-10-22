set(_retrieve_sources_
        retrieve.cxx)
set(_retrieve_headers_
        retrieve.h)


add_library(retrieve ${_retrieve_sources_})

target_link_libraries(retrieve ${PYTHON_LIBRARIES})

install (TARGETS retrieve DESTINATION bin)
install (FILES ${_retrieve_headers_} DESTINATION include)
