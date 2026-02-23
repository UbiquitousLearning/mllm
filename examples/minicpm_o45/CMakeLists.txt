add_executable(mllm-minicpm-o45-runner main.cpp)
target_link_libraries(mllm-minicpm-o45-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-minicpm-o45-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-minicpm-o45-runner-python main_python.cpp)
target_link_libraries(mllm-minicpm-o45-runner-python PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-minicpm-o45-runner-python PRIVATE ${MLLM_INCLUDE_DIR})
