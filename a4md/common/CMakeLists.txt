set(_a4md_cmn_sources_
        dataspaces_writer.cxx
        dataspaces_reader.cxx
        chunk_writer.cxx
        chunk_reader.cxx
        ims_reader.cxx
        ims_writer.cxx
        chunker.cxx
        py_runner.cxx
        chunk_serializer.cxx
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
        py_runner.h
        timer.h
        chunk_serializer.h
        )

add_library(a4md_cmn ${_a4md_cmn_sources_})
add_dependencies(a4md_cmn dataspaces)
set_target_properties(a4md_cmn PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# find Cray-specific modules
if (nersc)
    find_package(CRAY REQUIRED)
endif (nersc)

set(CMN_BUILD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CRAY_INCLUDE_DIRS})
set(CMN_INSTALL_INCLUDE_DIRS "include" ${CRAY_INCLUDE_DIRS})
target_include_directories(
    a4md_cmn PUBLIC
    # Headers used from source/build location:
    "$<BUILD_INTERFACE:${CMN_BUILD_INCLUDE_DIRS}>"
    # Headers used from the installed location:
    "$<INSTALL_INTERFACE:${CMM_INSTALL_INCLUDE_DIRS}>"
)
if (nersc)I
    target_link_libraries(a4md_cmn dataspaces dataspaces_cmn dataspaces_dart m rt ${CRAY_LDFLAGS} -dynamic ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${TAU_LIBRARIES})
else (nersc)
    target_link_libraries(a4md_cmn dataspaces dataspaces_cmn dataspaces_dart m rt ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${TAU_LIBRARIES})
endif (nersc)

add_subdirectory(test)
install (TARGETS a4md_cmn DESTINATION lib)
install (FILES ${_a4md_cmn_headers_} DESTINATION include)
