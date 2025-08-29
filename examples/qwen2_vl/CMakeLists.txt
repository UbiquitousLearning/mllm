add_executable(mllm-qwen2-vl-runner main.cpp)
target_link_libraries(mllm-qwen2-vl-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2-vl-runner PRIVATE ${MLLM_INCLUDE_DIR})
