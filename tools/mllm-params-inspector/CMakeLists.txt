add_executable(mllm-params-inspector main.cpp)
target_link_libraries(mllm-params-inspector PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-params-inspector PRIVATE ${MLLM_INCLUDE_DIR})
