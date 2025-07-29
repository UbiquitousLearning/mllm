add_executable(Mllm-Test-IR-TraceFooNet TraceFooNetTest.cpp)
target_link_libraries(Mllm-Test-IR-TraceFooNet PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-IR-TraceFooNet PRIVATE ${MLLM_INCLUDE_DIR})
