add_executable(mllm-ling3-runner main.cpp)
target_link_libraries(mllm-ling3-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-ling3-runner PRIVATE ${MLLM_INCLUDE_DIR})
