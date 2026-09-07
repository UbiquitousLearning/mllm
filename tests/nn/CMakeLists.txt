add_executable(Mllm-Test-Nn-FooNetTest FooNetTest.cpp)
target_link_libraries(Mllm-Test-Nn-FooNetTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-FooNetTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-StaticCacheTest StaticCacheTest.cpp)
target_link_libraries(Mllm-Test-Nn-StaticCacheTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-StaticCacheTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-KVHeadStaticCache KVHeadStaticCacheTest.cpp)
target_link_libraries(Mllm-Test-Nn-KVHeadStaticCache PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-KVHeadStaticCache PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-GroupedQueryAttention GroupedQueryAttentionTest.cpp)
target_link_libraries(Mllm-Test-Nn-GroupedQueryAttention PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-GroupedQueryAttention PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-CausalDepthwiseConv1D CausalDepthwiseConv1DTest.cpp)
target_link_libraries(Mllm-Test-Nn-CausalDepthwiseConv1D PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-CausalDepthwiseConv1D PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-ParallelLinear ParallelLinearTest.cpp)
target_link_libraries(Mllm-Test-Nn-ParallelLinear PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-ParallelLinear PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Nn-GatedDeltaRule GatedDeltaRuleTest.cpp)
target_link_libraries(Mllm-Test-Nn-GatedDeltaRule PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Nn-GatedDeltaRule PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
add_test(NAME GatedDeltaRuleFocused COMMAND Mllm-Test-Nn-GatedDeltaRule)
add_test(NAME GroupedQueryAttentionFocused COMMAND Mllm-Test-Nn-GroupedQueryAttention)
add_test(NAME KVHeadStaticCacheFocused COMMAND Mllm-Test-Nn-KVHeadStaticCache)
set_tests_properties(GroupedQueryAttentionFocused KVHeadStaticCacheFocused PROPERTIES LABELS nn-op)
