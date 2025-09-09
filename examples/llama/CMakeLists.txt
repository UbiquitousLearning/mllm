add_executable(mllm-tiny-llama-runner demo_tiny_llama.cpp)
target_link_libraries(mllm-tiny-llama-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-tiny-llama-runner PRIVATE ${MLLM_INCLUDE_DIR})