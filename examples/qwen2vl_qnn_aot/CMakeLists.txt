# AOT targets run on x86 and generate QNN context binaries for Android.
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-qwen2vl-aot-c compile.cpp)
  target_link_libraries(mllm-qwen2vl-aot-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen2vl-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen2vl-visual-aot-diag compile_visual.cpp)
  target_link_libraries(mllm-qwen2vl-visual-aot-diag PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen2vl-visual-aot-diag PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen2vl-visual-padding-diag visual_padding_diag.cpp)
  target_link_libraries(mllm-qwen2vl-visual-padding-diag PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen2vl-visual-padding-diag PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-qwen2vl-aot-runner aot_run.cpp)
target_link_libraries(mllm-qwen2vl-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen2vl-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2vl-visual-aot-runner visual_aot_run.cpp)
target_link_libraries(mllm-qwen2vl-visual-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen2vl-visual-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2vl-visual-padding-ab-diag visual_padding_ab_diag.cpp)
target_link_libraries(mllm-qwen2vl-visual-padding-ab-diag PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen2vl-visual-padding-ab-diag PRIVATE ${MLLM_INCLUDE_DIR})
