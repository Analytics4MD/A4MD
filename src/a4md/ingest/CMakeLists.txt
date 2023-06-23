set(_a4md_ingest_sources_
    src/ingester.cxx
    src/md_generator.cxx
    # src/chunk_stager.cxx
    # src/md_stager.cxx
    src/pdb_chunker.cxx
    src/cv_buffer.cxx
)
set(_a4md_ingest_headers_
    include/ingester.h
    include/md_generator.h
    # include/chunk_stager.h
    # include/md_stager.h
    include/pdb_chunker.h
    include/cv_buffer.h
)

blt_add_library(
    NAME a4md_ingest
    SOURCES ${_a4md_ingest_sources_}
    HEADERS ${_a4md_ingest_headers_}
    INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include>
    DEPENDS_ON a4md_cmn a4md_dtl boost mpi python_dev
)
set_target_properties(
    a4md_ingest
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if (ENABLE_TAU)
    blt_patch_target(
        NAME a4md_ingest
        DEPENDS_ON tau
    )
endif (ENABLE_TAU)

if (BUILD_TESTS)
    add_subdirectory(test)
endif (BUILD_TESTS)

install (TARGETS a4md_ingest DESTINATION lib)
install (FILES ${_a4md_ingest_headers_} DESTINATION include)
