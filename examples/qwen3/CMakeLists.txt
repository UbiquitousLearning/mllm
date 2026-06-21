add_executable(mllm-qwen3-runner main.cpp)
target_link_libraries(mllm-qwen3-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-runner PRIVATE ${MLLM_INCLUDE_DIR})
