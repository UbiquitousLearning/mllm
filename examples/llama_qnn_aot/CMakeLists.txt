# AOT targets run on x86
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-llama-aot-c compile.cpp)
  target_link_libraries(mllm-llama-aot-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-llama-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-llama-aot-c-sha compile_sha.cpp)
  target_link_libraries(mllm-llama-aot-c-sha PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-llama-aot-c-sha PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-llama-aot-runner aot_run.cpp)
target_link_libraries(mllm-llama-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-llama-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})
