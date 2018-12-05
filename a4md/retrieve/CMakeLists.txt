set(_retrieve_sources_
        retrieve.cxx)
set(_retrieve_headers_
        retrieve.h)

add_library(retrieve ${_retrieve_sources_})
set_target_properties(retrieve PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_dependencies(retrieve a4md_cmn)
target_link_libraries(retrieve a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES})

install (TARGETS retrieve DESTINATION lib)
install (FILES ${_retrieve_headers_} DESTINATION include)
