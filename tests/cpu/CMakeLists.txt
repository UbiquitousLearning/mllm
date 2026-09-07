add_executable(Mllm-Test-CPUKernel KernelTest.cpp)
target_link_libraries(Mllm-Test-CPUKernel PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-CPUKernel PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-KaiW4A32Pack KaiW4A32PackTest.cpp)
target_link_libraries(Mllm-Test-KaiW4A32Pack PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-KaiW4A32Pack PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-CPUContiguousOp ContiguousOpTest.cpp)
target_link_libraries(Mllm-Test-CPUContiguousOp PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-CPUContiguousOp PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
add_test(
  NAME CPUKernelFocused
  COMMAND
    Mllm-Test-CPUKernel
    --gtest_filter=CausalDepthwiseConvKernelTest.*:CausalDepthwiseConvCurrentFirstKernelTest.*:GatedDeltaRuleKernelTest.*:GqaDecodeKernelTest.*)
set_tests_properties(CPUKernelFocused PROPERTIES LABELS cpu-kernel)
