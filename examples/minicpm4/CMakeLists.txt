add_executable(mllm-minicpm4-runner main.cpp)
target_link_libraries(mllm-minicpm4-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-minicpm4-runner PRIVATE ${MLLM_INCLUDE_DIR})
