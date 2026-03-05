add_executable(mllm-qwen2_5-omni-text-runner text_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-text-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-text-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-image-runner image_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-image-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-image-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-audio-runner audio_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-audio-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-audio-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-audio-out-runner audio_out_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-audio-out-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-audio-out-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-image-runner-dbg image_infer_dbg.cpp)
target_link_libraries(mllm-qwen2_5-omni-image-runner-dbg PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-image-runner-dbg PRIVATE ${MLLM_INCLUDE_DIR})
