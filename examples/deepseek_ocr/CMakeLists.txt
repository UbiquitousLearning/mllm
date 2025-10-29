add_executable(mllm-deepseek-ocr-runner main.cpp)
target_link_libraries(mllm-deepseek-ocr-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-deepseek-ocr-runner PRIVATE ${MLLM_INCLUDE_DIR})
