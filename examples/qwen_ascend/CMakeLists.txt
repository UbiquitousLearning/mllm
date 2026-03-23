add_executable(mllm-qwen-ascend-runner main.cpp)
target_link_libraries(mllm-qwen-ascend-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen-ascend-runner PRIVATE ${MLLM_INCLUDE_DIR})

if(MLLM_BUILD_ASCEND_BACKEND)
  target_link_libraries(mllm-qwen-ascend-runner PRIVATE MllmAscendBackend)
  target_compile_definitions(mllm-qwen-ascend-runner PRIVATE MLLM_BUILD_ASCEND_BACKEND)
endif()

