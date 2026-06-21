# SPDX-License-Identifier: MIT
# Prefer the vendored CPM.cmake from the parent mllm repo. This avoids relying
# on network access for editable builds while keeping standalone fallback logic.

set(CPM_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_VERSION}.cmake")
set(PARENT_CPM "${CMAKE_CURRENT_LIST_DIR}/../../cmake/CPM.cmake")

if(EXISTS "${PARENT_CPM}")
  include("${PARENT_CPM}")
else()
  if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
    message(STATUS "Downloading CPM.cmake v${CPM_VERSION}...")
    file(DOWNLOAD
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake
      "${CPM_DOWNLOAD_LOCATION}"
      STATUS download_status
    )
    list(GET download_status 0 download_status_code)
    if(NOT download_status_code EQUAL 0)
      message(FATAL_ERROR "Failed to download CPM.cmake")
    endif()
  endif()

  include("${CPM_DOWNLOAD_LOCATION}")
endif()

if(NOT COMMAND CPMAddPackage)
  message(FATAL_ERROR "CPM.cmake loaded, but CPMAddPackage is not available")
endif()
