add_executable(mllm-internlm2_5-chat-runner main.cpp)
target_link_libraries(mllm-internlm2_5-chat-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-internlm2_5-chat-runner PRIVATE ${MLLM_INCLUDE_DIR})