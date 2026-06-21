add_executable(Mllm-Test-CPUKernel KernelTest.cpp)
target_link_libraries(Mllm-Test-CPUKernel PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-CPUKernel PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
