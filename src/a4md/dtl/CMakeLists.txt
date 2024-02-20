set(_a4md_dtl_sources_
    src/dataspaces_writer.cxx
    src/dataspaces_reader.cxx
    # src/decaf_writer.cxx
    # src/decaf_reader.cxx
    # src/chunk_writer.cxx
    # src/chunk_reader.cxx
    src/ims_reader.cxx
    src/ims_writer.cxx
    )
#include_directories(include/)
set(_a4md_dtl_headers_
    include/dataspaces_writer.h
    include/dataspaces_reader.h
    # include/decaf_writer.h
    # include/decaf_reader.h
    # include/chunk_writer.h
    # include/chunk_reader.h
    include/ims_writer.h
    include/ims_reader.h
    )
add_library(a4md_dtl ${_a4md_dtl_sources_})
target_include_directories(a4md_dtl PUBLIC include/)
set_target_properties(a4md_dtl PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
add_dependencies(a4md_dtl a4md_cmn)
target_link_libraries(a4md_dtl a4md_cmn)
if(TARGET_SYSTEM)
    set (TARGET_SYSTEM "${TARGET_SYSTEM}")
else()
    set(TARGET_SYSTEM "local")
endif()
message(STATUS "TARGET_SYSTEM is set to ${TARGET_SYSTEM}")
if(TARGET_SYSTEM STREQUAL "cori")
    message("The code is built at NERSC/" $ENV{NERSC_HOST})
    # Enable rounding up 8-byte allignment
    target_compile_definitions(-DNERSC=1)
    message(STATUS "NERSC build is enabled")
endif()

if (dtl_dspaces OR dtl_dimes)
    target_link_libraries(a4md_dtl dspaces::dspaces)
endif ()

if (dtl_dimes)
    message(STATUS "Building with DIMES data transport layer")
    add_definitions(-DDTL_DIMES)
endif ()

if (dtl_decaf)
    set(_decaf_dtl_sources_ decaf_writer.cxx decaf_reader.cxx)
    target_sources(a4md_dtl PRIVATE ${_decaf_dtl_sources_})
    set(_a4md_dtl_headers_ ${_a4md_dtl_headers_} decaf_writer.h decaf_reader.h)
endif ()

target_link_libraries(a4md_dtl ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${TAU_LIBRARIES} ${DTL_LIBRARIES})
get_target_property(A4MD_DTL_LIBRARIES a4md_dtl LINK_LIBRARIES)
message(STATUS "A4MD_DTL_LIBRARIES = ${A4MD_DTL_LIBRARIES}")

add_subdirectory(test)
install (TARGETS a4md_dtl DESTINATION lib)
install (FILES ${_a4md_dtl_headers_} DESTINATION include)
