include_directories(${MPI_C_INCLUDE_PATH})

set(WRAP ${PROJECT_SOURCE_DIR}/a4md/extern/wrap/wrap.py)
include(${PROJECT_SOURCE_DIR}/a4md/extern/wrap/WrapConfig.cmake)

add_wrapped_file(mpmd.C mpmd.w)
add_library(mpmd SHARED mpmd.C)
set_property(TARGET mpmd PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(mpmd ${MPI_C_LIBRARIES})
install (TARGETS mpmd DESTINATION lib)

