add_executable(mllm-qwen2_5-omni-text-runner text_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-text-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-text-runner PRIVATE ${MLLM_INCLUDE_DIR})
