add_executable(mllm-qwen2_5-omni-text-runner text_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-text-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-text-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-image-runner image_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-image-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-image-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen2_5-omni-audio-runner audio_infer.cpp)
target_link_libraries(mllm-qwen2_5-omni-audio-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen2_5-omni-audio-runner PRIVATE ${MLLM_INCLUDE_DIR})
