set(_a4md_cmn_sources_
        src/py_runner.cxx
        src/md_runner.cxx
        src/cv_runner.cxx
        src/chunk_serializer.cxx
        src/chunk_writer.cxx
        src/chunk_reader.cxx
        src/chunk_operator.cxx
        src/chunk_stager.cxx
        src/md_stager.cxx
        src/md_intermediator.cxx
        )
include_directories(include/)
set(_a4md_cmn_headers_
        include/chunk.h
        include/md_chunk.h
        include/cv_chunk.h
        include/serializable_chunk.h
        include/exceptions.h
        include/py_runner.h
        include/md_runner.h
        include/cv_runner.h
        include/timer.h
        include/chunk_serializer.h
        include/chunk_writer.h
        include/chunk_reader.h
        include/chunk_operator.h
        include/chunk_stager.h
        include/md_stager.h
        include/md_intermediator.h
        )

add_library(a4md_cmn ${_a4md_cmn_sources_})
target_include_directories(a4md_cmn PUBLIC include/)
set_target_properties(a4md_cmn PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES})
get_target_property(A4MD_CMN_LIBRARIES a4md_cmn LINK_LIBRARIES)
message(STATUS "A4MD_CMN_LIBRARIES = ${A4MD_CMN_LIBRARIES}")

add_subdirectory(test)
install (TARGETS a4md_cmn DESTINATION lib)
install (FILES ${_a4md_cmn_headers_} DESTINATION include)
