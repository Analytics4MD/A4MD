set(_a4md_cmn_sources_
        dataspaces_writer.cxx
        chunk_writer.cxx
        chunk_reader.cxx
        ims_reader.cxx
        )
set(_a4md_cmn_headers_
        dataspaces_writer.h
        ims_writer.h
        chunk_writer.h
        chunk_reader.h
        ims_reader.h
        chunk.h
        exceptions.h
        )

add_library(a4md_cmn ${_a4md_cmn_sources_})
add_dependencies(a4md_cmn dataspaces)

target_include_directories(
    a4md_cmn PUBLIC
    # Headers used from source/build location:
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    # Headers used from the installed location:
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(a4md_cmn dataspaces dataspaces_cmn dataspaces_dart m ${Boost_LIBRARIES} ${MPI_LIBRARIES})

install (FILES ${_a4md_cmn_headers_} DESTINATION include)
