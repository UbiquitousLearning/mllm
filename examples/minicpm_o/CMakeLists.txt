add_executable(mllm-minicpm-o main.cpp)
target_link_libraries(mllm-minicpm-o PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-minicpm-o PRIVATE ${MLLM_INCLUDE_DIR})
