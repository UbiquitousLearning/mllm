# Copyright (c) MLLM Team.
# Licensed under the MIT License.

if(MLLM_BUILD_QNN_BACKEND)
  add_executable(QNNOutputOrderTest QNNOutputOrderTest.cpp)
  target_link_libraries(QNNOutputOrderTest PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(QNNOutputOrderTest PRIVATE ${MLLM_INCLUDE_DIR})
endif()

