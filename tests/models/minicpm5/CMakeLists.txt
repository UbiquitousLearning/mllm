add_executable(Mllm-Test-MiniCPM5-Config MiniCPM5ConfigTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Config PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Config PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-MiniCPM5-Config
                           PRIVATE MINICPM5_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/minicpm5")

add_executable(Mllm-Test-MiniCPM5-Tokenizer MiniCPM5TokenizerTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Tokenizer PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Tokenizer PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-MiniCPM5-Model MiniCPM5ModelTest.cpp)
target_link_libraries(Mllm-Test-MiniCPM5-Model PRIVATE gtest_main MllmCPUBackend)
target_include_directories(Mllm-Test-MiniCPM5-Model PRIVATE ${MLLM_INCLUDE_DIR})
target_compile_definitions(Mllm-Test-MiniCPM5-Model
                           PRIVATE MINICPM5_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/minicpm5")

add_test(NAME MiniCPM5ConfigFocused COMMAND Mllm-Test-MiniCPM5-Config)
add_test(NAME MiniCPM5TokenizerFocused COMMAND Mllm-Test-MiniCPM5-Tokenizer)
add_test(NAME MiniCPM5ModelFocused COMMAND Mllm-Test-MiniCPM5-Model)
set_tests_properties(MiniCPM5ConfigFocused MiniCPM5TokenizerFocused MiniCPM5ModelFocused PROPERTIES LABELS minicpm5)
