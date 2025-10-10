add_executable(mllm-qwen3-service main.cpp)
target_link_libraries(mllm-qwen3-service PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-service PRIVATE ${MLLM_INCLUDE_DIR})
