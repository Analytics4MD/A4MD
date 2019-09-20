######################################################
# - Try to find drc
# Once done this will define
#  CRAY_FOUND - System has CRAY modules
#  CRAY_INCLUDE_DIRS - The CRAY module include directories
#  CRAY_LIBRARIES - The libraries needed to use CRAY modules
#  CRAY_CFLAGS - The compiler flags in compiling
#  CRAY_LDFLAGS - The library flags in linking

######################################################

find_package(PkgConfig)
pkg_check_modules(CRAY cray-drc cray-ugni cray-pmi cray-gni-headers cray-job cray-jobctl)
if (CRAY_FOUND)
    #message(STATUS "CRAY_INCLUDE_DIRS = " ${CRAY_INCLUDE_DIRS})
    #message(STATUS "CRAY_LIBRARY_DIR = " ${CRAY_LIBRARY_DIRS})
    #message(STATUS "CRAY_LIBRARIES = " ${CRAY_LIBRARIES})
    message(STATUS "CRAY_LDFLAGS = " ${CRAY_LDFLAGS})
    message(STATUS "CRAY_CFLAGS = " ${CRAY_CFLAGS})
endif()
