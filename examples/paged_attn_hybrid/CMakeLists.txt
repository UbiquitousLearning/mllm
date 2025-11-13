add_executable(mllm-paged-attn-hybrid main.cpp)
target_link_libraries(mllm-paged-attn-hybrid PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-paged-attn-hybrid PRIVATE ${MLLM_INCLUDE_DIR})
