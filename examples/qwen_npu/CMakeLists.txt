add_executable(mllm-qwen-npu main.cpp)
target_link_libraries(mllm-qwen-npu PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen-npu PRIVATE ${MLLM_INCLUDE_DIR})