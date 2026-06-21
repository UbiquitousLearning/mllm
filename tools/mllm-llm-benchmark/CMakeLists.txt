add_executable(mllm-llm-benchmark main.cpp)
target_link_libraries(mllm-llm-benchmark PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-llm-benchmark PRIVATE ${MLLM_INCLUDE_DIR})
