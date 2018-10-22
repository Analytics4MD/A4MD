# adding python dependency 
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# add libraries
include_directories("${PROJECT_SOURCE_DIR}/a4md/ingest")
include_directories("${PROJECT_SOURCE_DIR}/a4md/retrieve")
add_subdirectory(ingest)
add_subdirectory(retrieve)
add_subdirectory(extern)


# add project level sources
set(_a4md_sources_
        main.cxx)


# add executables
add_executable(a4md ${_a4md_sources_})
add_executable(ingester ingester.cxx)
add_executable(retriever retriever.cxx)
target_link_libraries(ingester ingest)
target_link_libraries(retriever retrieve)
