# Copied from
# https://github.com/Analytics4MD/A4MD-sample-workflow

# Find A4MD include directories and libraries
# A4MD_INCLUDE_DIRECTORIES - where to find A4MD headers
# A4MD_LIBRARIES           - list of libraries to link against when using A4MD
# A4MD_FOUND               - Do not attempt to use A4MD if "no", "0", or undefined.

include(FindPackageHandleStandardArgs)

find_path(A4MD_INCLUDE_DIRS NAMES chunk.h HINTS
        ${A4MD_PREFIX}/include
        /usr/include
        /usr/local/include
        /opt/local/include
        /sw/include
)

find_library(A4MD_INGEST_LIBRARIES NAMES a4md_ingest HINTS
        ${A4MD_PREFIX}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
)

find_library(A4MD_RETRIEVE_LIBRARIES NAMES a4md_retrieve HINTS
        ${A4MD_PREFIX}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
)

find_library(A4MD_DTL_LIBRARIES NAMES a4md_dtl HINTS
        ${A4MD_PREFIX}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
)

find_library(A4MD_COMMON_LIBRARIES NAMES a4md_cmn HINTS
        ${A4MD_PREFIX}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
)

set(A4MD_LIBRARIES ${A4MD_INGEST_LIBRARIES} ${A4MD_RETRIEVE_LIBRARIES} ${A4MD_DTL_LIBRARIES} ${A4MD_COMMON_LIBRARIES})

SET(A4MD_TRANSPORT_LIBRARIES "")

find_package(dspaces CONFIG REQUIRED)
set(A4MD_INCLUDE_DIRS ${A4MD_INCLUDE_DIRS})
set(A4MD_TRANSPORT_LIBRARIES ${A4MD_TRANSPORT_LIBRARIES} dspaces::dspaces)
message(STATUS "${A4MD_TRANSPORT_LIBRARIES}")
list(APPEND A4MD_LIBRARIES ${A4MD_TRANSPORT_LIBRARIES})

find_package_handle_standard_args(A4MD DEFAULT_MSG
        A4MD_INCLUDE_DIRS
        A4MD_INGEST_LIBRARIES
        A4MD_RETRIEVE_LIBRARIES
        A4MD_DTL_LIBRARIES
        A4MD_COMMON_LIBRARIES
        A4MD_TRANSPORT_LIBRARIES
)

mark_as_advanced(
        A4MD_INCLUDE_DIRS
        A4MD_INGEST_LIBRARIES
        A4MD_RETRIEVE_LIBRARIES
        A4MD_DTL_LIBRARIES
        A4MD_COMMON_LIBRARIES
        A4MD_TRANSPORT_LIBRARIES
        A4MD_LIBRARIES
        A4MD_FOUND
)