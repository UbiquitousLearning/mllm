add_executable(Mllm-Test-Nn-FooNetTest FooNetTest.cpp)
target_link_libraries(Mllm-Test-Nn-FooNetTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-FooNetTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-StaticCacheTest StaticCacheTest.cpp)
target_link_libraries(Mllm-Test-Nn-StaticCacheTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-StaticCacheTest PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
