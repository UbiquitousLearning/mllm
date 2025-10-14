add_executable(Mllm-Test-CUDA-DeviceInfo DeviceInfoTest.cpp)
target_link_libraries(Mllm-Test-CUDA-DeviceInfo PRIVATE MllmRT MllmCPUBackend MllmCUDABackend)
target_include_directories(Mllm-Test-CUDA-DeviceInfo PRIVATE ${MLLM_INCLUDE_DIR})
