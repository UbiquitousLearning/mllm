# Copyright (c) MLLM Team.
# Licensed under the MIT License.

# Ascend Graph Execution Module
# This module provides graph compilation and execution support for Ascend backend

file(GLOB MLLM_ASCEND_GRAPH_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(MllmAscendGraph OBJECT ${MLLM_ASCEND_GRAPH_FILES})

# Include directories using environment variables (matching parent CMakeLists.txt)
target_include_directories(MllmAscendGraph PUBLIC
  ${CMAKE_SOURCE_DIR}
)

if(DEFINED ENV{ASCEND_HOME_PATH})
  target_include_directories(MllmAscendGraph PUBLIC $ENV{ASCEND_HOME_PATH}/include)
endif()

if(DEFINED ENV{ATB_HOME_PATH})
  target_include_directories(MllmAscendGraph PUBLIC $ENV{ATB_HOME_PATH}/include)

  # Keep graph builder ABI consistent with ATB cxx_abi_0 package.
  if("$ENV{ATB_HOME_PATH}" MATCHES "cxx_abi_0")
    target_compile_definitions(MllmAscendGraph PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
  endif()
endif()

target_link_libraries(MllmAscendGraph PUBLIC
  MllmRT
  ascendcl
  atb
)

# Ensure C++20 standard
set_target_properties(MllmAscendGraph PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
