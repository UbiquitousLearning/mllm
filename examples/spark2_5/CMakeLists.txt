add_executable(mllm-spark25-runner main.cpp)
target_link_libraries(mllm-spark25-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-spark25-runner PRIVATE ${MLLM_INCLUDE_DIR})
