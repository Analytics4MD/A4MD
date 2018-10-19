
add_subdirectory(extern)

set(_a4md_sources_
        ingester.cxx
        retriever.cxx
        main.cxx)
set(_a4md_headers_
        ingester.h
        retriever.h)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(a4md ${_a4md_sources_})

target_link_libraries(a4md ${PYTHON_LIBRARIES})
