set(_a4md_retrieve_sources_
    src/md_analyzer.cxx
    src/chunk_analyzer.cxx
    src/md_retriever.cxx
    src/retriever.cxx
)
set(_a4md_retrieve_headers_
    include/md_analyzer.h
    include/chunk_analyzer.h
    include/md_retriever.h
    include/retriever.h
)

blt_add_library(
    NAME a4md_retrieve
    SOURCES ${_a4md_retrieve_sources_}
    HEADERS ${_a4md_retrieve_headers_}
    INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include>
    DEPENDS_ON a4md_cmn a4md_dtl boost mpi python_dev
)
set_target_properties(
    a4md_retrieve
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if (ENABLE_TAU)
    blt_patch_target(
        NAME a4md_retrieve
        DEPENDS_ON tau
    )
endif (ENABLE_TAU)

if (BUILD_TESTS)
    add_subdirectory(test)
endif (BUILD_TESTS)

install (TARGETS a4md_retrieve DESTINATION lib)
install (FILES ${_a4md_retrieve_headers_} DESTINATION include)
