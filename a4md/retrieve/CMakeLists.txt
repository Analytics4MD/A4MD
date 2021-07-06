set(_a4md_retrieve_sources_
        md_analyzer.cxx
        # chunk_analyzer.cxx
        md_retriever.cxx
        retriever.cxx
		)
set(_a4md_retrieve_headers_
        md_analyzer.h
        # chunk_analyzer.h
        md_retriever.h
        retriever.h
		)

add_library(a4md_retrieve ${_a4md_retrieve_sources_})
set_target_properties(a4md_retrieve PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_dependencies(a4md_retrieve a4md_cmn a4md_dtl)
target_link_libraries(a4md_retrieve a4md_cmn a4md_dtl ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES})
get_target_property(A4MD_RETRIEVE_LIBRARIES a4md_retrieve LINK_LIBRARIES)
message(STATUS "A4MD_RETRIEVE_LIBRARIES = ${A4MD_RETRIEVE_LIBRARIES}")

add_subdirectory(test)
install (TARGETS a4md_retrieve DESTINATION lib)
install (FILES ${_a4md_retrieve_headers_} DESTINATION include)
