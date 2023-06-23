set(_a4md_dtl_sources_
    src/ims_reader.cxx
    src/ims_writer.cxx
)
set(_a4md_dtl_headers_
    include/ims_reader.h
    include/ims_writer.h
)

if (ENABLE_DECAF)
    list(APPEND _a4md_dtl_sources_
        src/decaf_writer.cxx
        src/decaf_reader.cxx
    )
    list(APPEND _a4md_dtl_headers_
        include/decaf_writer.h
        include/decaf_reader.h
    )
endif (ENABLE_DECAF)

blt_add_library(
    NAME a4md_dtl
    SOURCES ${_a4md_dtl_sources_}
    HEADERS ${_a4md_dtl_headers_}
    INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
             $<INSTALL_INTERFACE:include>
    DEPENDS_ON a4md_cmn boost mpi
)
set_target_properties(
    a4md_dtl
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if (ENABLE_DSPACES)
    blt_add_library(
        NAME a4md_dtl_dspaces
        SOURCES src/dataspaces_writer.cxx
                src/dataspaces_reader.cxx
        HEADERS include/dataspaces_writer.h
                include/dataspaces_reader.h
        OBJECT TRUE
    )
    if (DATASPACES_FOUND)
        blt_patch_target(
            NAME a4md_dtl_dspaces
            DEPENDS_ON dataspaces
            INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>
        )
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
    
        blt_import_library(
            NAME dataspaces
            DEPENDS_ON project_dataspaces
            LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdspaces.a
                      ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdscommon.a
                      ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/lib/libdart.a
            INCLUDES ${CMAKE_CURRENT_BINARY_DIR}/extern/dataspaces/include
            EXPORTABLE ON
        )
        set_target_properties(
            dataspaces
            PROPERTIES POSITION_INDEPENDENT_CODE ON
        )
    
        blt_patch_target(
            NAME a4md_dtl_dspaces
            DEPENDS_ON dataspaces
        )

        # find Cray-specific modules
        if (TARGET_SYSTEM STREQUAL "cori")
            find_package(CRAY REQUIRED)
            set(DTL_BUILD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CRAY_INCLUDE_DIRS})
            set(DTL_INSTALL_INCLUDE_DIRS "include" ${CRAY_INCLUDE_DIRS})
            blt_patch_target(
                NAME a4md_dtl_dspaces
                INCLUDES $<BUILD_INTERFACE:${DTL_BUILD_INCLUDE_DIRS}>
                         $<INSTALL_INTERFACE:${DTL_INSTALL_INCLUDE_DIRS}>
                LIBRARIES m ${CRAY_LIBRARIES} # Tu's code also depended on rt, but that shouldn't be needed
                LINK_FLAGS ${CRAY_LDFLAGS} -dynamic
            )
        elseif ((TARGET_SYSTEM STREQUAL "tellico") OR (TARGET_SYSTEM STREQUAL "summit"))
            find_package(ibverbs REQUIRED)
            find_package(rdmacm REQUIRED)
            set(
                DTL_BUILD_INCLUDE_DIRS
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${RDMACM_INCLUDE_DIR}
                ${IBVERBS_INCLUDE_DIRS}
            )
            set(
                DTL_INSTALL_INCLUDE_DIRS
                include
                ${RDMACM_INCLUDE_DIR}
                ${IBVERBS_INCLUDE_DIRS}
            )
            blt_patch_target(
                NAME a4md_dtl_dspaces
                INCLUDES $<BUILD_INTERFACE:${DTL_BUILD_INCLUDE_DIRS}>
                         $<INSTALL_INTERFACE:${DTL_INSTALL_INCLUDE_DIRS}
                LIBRARIES ${RDMACM_LIBRARIES} ${IBVERBS_LIBRARIES}
                LINK_FLAGS -dynamic
            )
        endif ()
    endif()
    blt_patch_target(
        NAME a4md_dtl
        DEPENDS_ON a4md_dtl_dspaces
    )
endif (ENABLE_DSPACES)

if (ENABLE_DIMES)
    message(STATUS "Building with DIMES data transport layer")
    add_definitions(-DDTL_DIMES)
endif (ENABLE_DIMES)

if (dtl_decaf)
    blt_add_library(
        NAME a4md_dtl_decaf
        SOURCES src/decaf_writer.cxx
                src/decaf_reader.cxx
        INCLUDES include/decaf_writer.h
                 include/decaf_reader.h
        DEPENDS_ON decaf
        OBJECT TRUE
    )
    blt_patch_target(
        NAME a4md_dtl
        DEPENDS_ON a4md_dtl_decaf
    )
endif ()

if (ENABLE_TAU)
    blt_patch_target(
        NAME a4md_dtl
        DEPENDS_ON tau
    )
endif (ENABLE_TAU)

if (BUILD_TESTS)
    add_subdirectory(test)
endif (BUILD_TESTS)

install (TARGETS a4md_dtl DESTINATION lib)
# Try replacing with PUBLIC_HEADER on a4md_dtl
install (FILES ${_a4md_dtl_headers_} DESTINATION include)
