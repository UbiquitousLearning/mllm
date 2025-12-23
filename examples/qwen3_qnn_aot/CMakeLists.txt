add_executable(mllm-qwen3-aot-c compile.cpp)
target_link_libraries(mllm-qwen3-aot-c PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

target_include_directories(mllm-qwen3-aot-c PRIVATE
  $ENV{QAIRT_SDK_ROOT}/include # QNN SDK include
  $ENV{QAIRT_SDK_ROOT}/include/QNN # QNN SDK include
)
