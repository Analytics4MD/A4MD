set(_retrieve_sources_
        retrieve.cxx)
set(_retrieve_headers_
        retrieve.h)


add_library(retrieve ${_retrieve_sources_})
set_property(TARGET retrieve PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(retrieve ${PYTHON_LIBRARIES})

install (TARGETS retrieve DESTINATION lib)
install (FILES ${_retrieve_headers_} DESTINATION include)
