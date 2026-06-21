add_executable(Mllm-Test-Core-LoadParams LoadParamsTest.cpp)
target_link_libraries(Mllm-Test-Core-LoadParams PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Core-LoadParams PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
