set(_a4md_cmn_sources_
        chunk.cxx
        py_caller.cxx
        py_runner.cxx
        md_runner.cxx
        cv_runner.cxx
        chunk_serializer.cxx
        chunk_writer.cxx
        chunk_reader.cxx
        chunk_operator.cxx
        chunk_stager.cxx
        # md_stager.cxx
        md_intermediator.cxx
        )
set(_a4md_cmn_headers_
        chunk.h
        md_chunk.h
        cv_chunk.h
        chunks.h
        serializable_chunk.h
        exceptions.h
        py_caller.h
        py_runner.h
        md_runner.h
        cv_runner.h
        timer.h
        chunk_serializer.h
        chunk_writer.h
        chunk_reader.h
        chunk_operator.h
        chunk_stager.h
        # md_stager.h
        md_intermediator.h
        )
add_library(a4md_cmn ${_a4md_cmn_sources_})
set_target_properties(a4md_cmn PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(a4md_cmn ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${PYTHON_LIBRARIES} ${TAU_LIBRARIES} pybind11::module pybind11::lto pybind11::embed)
get_target_property(A4MD_CMN_LIBRARIES a4md_cmn LINK_LIBRARIES)
message(STATUS "A4MD_CMN_LIBRARIES = ${A4MD_CMN_LIBRARIES}")

# A4MD pybind11 module
set(_py_a4md_sources_ 
    py_chunks.cxx 
    chunk.cxx
    )
set(_py_a4md_headers_ 
    py_chunks.h
    )
pybind11_add_module(pya4md ${_py_a4md_sources_})

# Common tests
add_subdirectory(test)
install (TARGETS a4md_cmn DESTINATION lib)
install (FILES ${_a4md_cmn_headers_} DESTINATION include)
install (TARGETS pya4md DESTINATION lib)
