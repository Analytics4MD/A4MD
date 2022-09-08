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

if (dtl_dspaces)
    if (DATASPACES_FOUND)
        install(DIRECTORY "${DATASPACES_ROOT}/bin/" 
            DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
            USE_SOURCE_PERMISSIONS
            PATTERN "test_*" EXCLUDE)
    else()
        # Check TARGET_SYSTEM predefined macros
        # Todo: This macro should be removed later 
        message(STATUS "Build Dataspaces internally")
        if(TARGET_SYSTEM)
            set (TARGET_SYSTEM "${TARGET_SYSTEM}")
        else()
            set(TARGET_SYSTEM "local")
        endif()
        message(STATUS "TARGET_SYSTEM is set to ${TARGET_SYSTEM}")
        if(TARGET_SYSTEM STREQUAL "cori")
            message("The code is built at NERSC/" $ENV{NERSC_HOST})
            # Enable rounding up 8-byte allignment
            add_definitions(-DNERSC)
            message(STATUS "NERSC build is enabled")
        endif()

        # adding Dataspaces external project
        message(STATUS "Adding Dataspaces library to project")
        include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

        if((TARGET_SYSTEM STREQUAL "summit") OR (TARGET_SYSTEM STREQUAL "tellico"))
            message(STATUS "Dataspaces on Summit/Tellico is having issue with Infiniband, temporarily use only --enable-shmem in the meantime.")
            set(DS_CONF_ENABLES "--enable-shmem")
        elseif(TARGET_SYSTEM STREQUAL "cori")
            message(STATUS "Dataspaces requires --enable-drc on Cori")
            set(DS_CONF_ENABLES "--enable-drc;--enable-dimes;--enable-shmem")
        else()
            set(DS_CONF_ENABLES "--enable-dart-tcp")
            # ToDo: Fix MPI_FORTRAN not found to unhardcode this
            set(MPI_FORTRAN_COMPILER "mpif90")
        endif()
        message(STATUS "DS_CONF_ENABLES ${DS_CONF_ENABLES}")

        # External project to build Dataspaces
        message(STATUS "Buiding Dataspaces")
        ExternalProject_Add(project_dataspaces
          SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/a4md/extern/dataspaces
          BUILD_IN_SOURCE 1
          CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure CC=${MPI_C_COMPILER} CXX=${MPI_CXX_COMPILER} FC=${MPI_FORTRAN_COMPILER} CFLAGS=-fPIC ${DS_CONF_ENABLES} --prefix=${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces
          BUILD_COMMAND ${MAKE}
        )

        # Add Dataspaces libraries to 
        add_library(dataspaces STATIC IMPORTED)
        add_dependencies(dataspaces project_dataspaces)
        set_property(TARGET dataspaces PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdspaces.a)
        set_target_properties(dataspaces PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            POSITION_INDEPENDENT_CODE ON
        )
        add_library(dataspaces_cmn STATIC IMPORTED)
        add_dependencies(dataspaces_cmn project_dataspaces)
        set_property(TARGET dataspaces_cmn PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdscommon.a)
        set_target_properties(dataspaces_cmn PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            POSITION_INDEPENDENT_CODE ON
        )
        add_library(dataspaces_dart STATIC IMPORTED)
        add_dependencies(dataspaces_dart project_dataspaces)
        set_property(TARGET dataspaces_dart PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdart.a)
        set_target_properties(dataspaces_dart PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            POSITION_INDEPENDENT_CODE ON
        )
        add_dependencies(a4md_dtl dataspaces dataspaces_cmn dataspaces_dart)
        include_directories("${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include")

        # Copy to installation directory
        install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/bin/" 
                DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
                USE_SOURCE_PERMISSIONS
                PATTERN "test_*" EXCLUDE)
        install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/" 
                DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
                USE_SOURCE_PERMISSIONS)
        install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include/" 
                DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
                USE_SOURCE_PERMISSIONS)

        # find Cray-specific modules
        if (TARGET_SYSTEM STREQUAL "cori")
            find_package(CRAY REQUIRED)
        endif()
        set(DTL_BUILD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CRAY_INCLUDE_DIRS})
        set(DTL_INSTALL_INCLUDE_DIRS "include" ${CRAY_INCLUDE_DIRS})
        target_include_directories(
            a4md_dtl PUBLIC
            # Headers used from source/build location:
            "$<BUILD_INTERFACE:${DTL_BUILD_INCLUDE_DIRS}>"
            # Headers used from the installed location:
            "$<INSTALL_INTERFACE:${DTL_INSTALL_INCLUDE_DIRS}>"
        )
        target_link_libraries(a4md_dtl dataspaces dataspaces_cmn dataspaces_dart m rt)
        if (TARGET_SYSTEM STREQUAL "cori")
            target_link_libraries(a4md_dtl ${CRAY_LDFLAGS} -dynamic)
        elseif ((TARGET_SYSTEM STREQUAL "tellico") OR (TARGET_SYSTEM STREQUAL "summit"))
            target_link_libraries(a4md_dtl ibverbs rdmacm -dynamic)
        endif ()
    endif()
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
