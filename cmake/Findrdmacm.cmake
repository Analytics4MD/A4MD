# This CMake file was obtained from SKAO/cmake-modules on GitLab
# All rights reserved by the original authors
# The copyright for this file is listed below:
#
# Copyright (c) 2022, SKA Organisation
# Copyright (c) 2022, CSIRO
# Copyright (c) 2022, The University of Western Australia (ICRAR)
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
#
# - Find rdma cm
# Find the rdma cm library and includes
#
# RDMACM_INCLUDE_DIR - where to find cma.h, etc.
# RDMACM_LIBRARIES - List of libraries when using rdmacm.
# RDMACM_FOUND - True if rdmacm found.

find_path(RDMACM_INCLUDE_DIR rdma/rdma_cma.h)
find_library(RDMACM_LIBRARIES rdmacm)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rdmacm DEFAULT_MSG RDMACM_LIBRARIES RDMACM_INCLUDE_DIR)

if(RDMACM_FOUND)
  if(NOT TARGET RDMA::RDMAcm)
    add_library(RDMA::RDMAcm UNKNOWN IMPORTED)
  endif()
  set_target_properties(RDMA::RDMAcm PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${RDMACM_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${RDMACM_LIBRARIES}")
endif()

mark_as_advanced(
  RDMACM_LIBRARIES
)