add_executable(Mllm-Test-AsyncIO-CpuMemDiskAsyncIo CpuMemDiskAsyncIoTest.cpp)
target_link_libraries(Mllm-Test-AsyncIO-CpuMemDiskAsyncIo PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-AsyncIO-CpuMemDiskAsyncIo PRIVATE ${MLLM_INCLUDE_DIR})
