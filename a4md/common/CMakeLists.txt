set(_a4md_cmn_sources_
        py_runner.cxx
        md_runner.cxx
        chunk_serializer.cxx
        chunk_writer.cxx
        chunk_reader.cxx
        chunk_stager.cxx
        md_stager.cxx
        )
set(_a4md_cmn_headers_
        chunk.h
        md_chunk.h
        serializable_chunk.h
        exceptions.h
        py_runner.h
        md_runner.h
        timer.h
        chunk_serializer.h
        chunk_writer.h
        chunk_reader.h
        chunk_stager.h
        md_stager.h
        )

add_library(a4md_cmn ${_a4md_cmn_sources_})
set_target_properties(a4md_cmn PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${TAU_LIBRARIES})
get_target_property(A4MD_CMN_LIBRARIES a4md_cmn LINK_LIBRARIES)
message(STATUS "A4MD_CMN_LIBRARIES = ${A4MD_CMN_LIBRARIES}")

add_subdirectory(test)
install (TARGETS a4md_cmn DESTINATION lib)
install (FILES ${_a4md_cmn_headers_} DESTINATION include)
