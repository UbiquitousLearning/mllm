add_executable(mllm-smollm3-runner main.cpp)

target_link_libraries(mllm-smollm3-runner PRIVATE 
    MllmRT 
    MllmCPUBackend
)

target_include_directories(mllm-smollm3-runner PRIVATE 
    ${MLLM_INCLUDE_DIR}
)

# Android 特定设置
if(ANDROID)
    target_link_libraries(mllm-smollm3-runner PRIVATE 
        log
        android
    )
endif()
