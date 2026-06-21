add_executable(mllm-qwen3-service main.cpp)
target_link_libraries(mllm-qwen3-service PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-service PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-accuracy test_accuracy.cpp)
target_link_libraries(mllm-qwen3-accuracy PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-accuracy PRIVATE ${MLLM_INCLUDE_DIR})


add_executable(mllm-qwen3-probing main_probing.cpp)
target_link_libraries(mllm-qwen3-probing PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-probing PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-trivia-probing test_trivia_probing.cpp)
target_link_libraries(mllm-qwen3-trivia-probing PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-trivia-probing PRIVATE ${MLLM_INCLUDE_DIR})