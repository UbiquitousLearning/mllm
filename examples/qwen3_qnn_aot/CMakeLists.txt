add_executable(mllm-qwen3-aot-c compile.cpp)
target_link_libraries(mllm-qwen3-aot-c PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-aot-c PRIVATE ${MLLM_INCLUDE_DIR})
