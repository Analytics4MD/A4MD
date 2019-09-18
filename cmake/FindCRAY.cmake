######################################################
# - Try to find drc
# Once done this will define
#  DRC_FOUND - System has DRC
#  DRC_INCLUDE_DIRS - The DRC include directories
#  DRC_LIBRARIES - The libraries needed to use DRC

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
