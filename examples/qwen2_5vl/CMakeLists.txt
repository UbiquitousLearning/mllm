add_executable(mllm-qwen2_5-vl-runner main.cpp)
target_link_libraries(mllm-qwen2_5-vl-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-vl-runner PRIVATE ${MLLM_INCLUDE_DIR})
