add_executable(mllm-qwen2_5-vl-tracer main.cpp)
target_link_libraries(mllm-qwen2_5-vl-tracer PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-vl-tracer PRIVATE ${MLLM_INCLUDE_DIR})
