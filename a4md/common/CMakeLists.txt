set(_a4md_cmn_sources_
        dataspaces_writer.cxx
        dataspaces_reader.cxx
        chunk_writer.cxx
        chunk_reader.cxx
        ims_reader.cxx
        ims_writer.cxx
        chunker.cxx
        )
set(_a4md_cmn_headers_
        dataspaces_writer.h
        dataspaces_reader.h
        chunk_writer.h
        chunk_reader.h
        ims_writer.h
        ims_reader.h
        chunker.h
        chunk.h
        exceptions.h
        )

add_library(a4md_cmn ${_a4md_cmn_sources_})
add_dependencies(a4md_cmn dataspaces)
set_target_properties(a4md_cmn PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    a4md_cmn PUBLIC
    # Headers used from source/build location:
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    # Headers used from the installed location:
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(a4md_cmn dataspaces dataspaces_cmn dataspaces_dart m ${Boost_LIBRARIES} ${MPI_LIBRARIES})

install (TARGETS a4md_cmn DESTINATION lib)
install (FILES ${_a4md_cmn_headers_} DESTINATION include)
