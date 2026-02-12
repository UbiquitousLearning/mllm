# SPDX-License-Identifier: MIT
# Download CPM.cmake on-the-fly
# This is a lightweight bootstrap that downloads the actual CPM.cmake

set(CPM_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_VERSION}.cmake")

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  message(STATUS "Downloading CPM.cmake v${CPM_VERSION}...")
  file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION}
    STATUS download_status
  )
  list(GET download_status 0 download_status_code)
  if(NOT download_status_code EQUAL 0)
    # Fallback: copy from parent mllm project if available
    set(PARENT_CPM "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake")
    if(EXISTS ${PARENT_CPM})
      message(STATUS "Using CPM.cmake from parent project")
      file(COPY ${PARENT_CPM} DESTINATION "${CMAKE_BINARY_DIR}/cmake/")
      file(RENAME "${CMAKE_BINARY_DIR}/cmake/CPM.cmake" ${CPM_DOWNLOAD_LOCATION})
    else()
      message(FATAL_ERROR "Failed to download CPM.cmake")
    endif()
  endif()
endif()

include(${CPM_DOWNLOAD_LOCATION})
