add_executable(Mllm-Test-Engine-PerfMem PerfMemTest.cpp)
target_link_libraries(Mllm-Test-Engine-PerfMem PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-PerfMem PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Engine-AsyncModule AsyncModuleTest.cpp)
target_link_libraries(Mllm-Test-Engine-AsyncModule PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Engine-AsyncModule PRIVATE ${MLLM_INCLUDE_DIR})
