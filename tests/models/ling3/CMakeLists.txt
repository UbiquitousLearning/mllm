add_executable(Mllm-Test-Ling3-Config Ling3ConfigTest.cpp)
target_link_libraries(Mllm-Test-Ling3-Config PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Ling3-Config PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-Ling3-Config
                           PRIVATE LING3_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/ling3")

add_executable(Mllm-Test-Ling3-Tokenizer Ling3TokenizerTest.cpp)
target_link_libraries(Mllm-Test-Ling3-Tokenizer PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Ling3-Tokenizer PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Ling3-RoPE Ling3RoPETest.cpp)
target_link_libraries(Mllm-Test-Ling3-RoPE PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-Ling3-RoPE PRIVATE ${MLLM_INCLUDE_DIR})

add_test(NAME Ling3ConfigFocused COMMAND Mllm-Test-Ling3-Config)
add_test(NAME Ling3TokenizerFocused COMMAND Mllm-Test-Ling3-Tokenizer)
add_test(NAME Ling3RoPEFocused COMMAND Mllm-Test-Ling3-RoPE)
set_tests_properties(Ling3ConfigFocused Ling3TokenizerFocused Ling3RoPEFocused PROPERTIES LABELS ling3)
