cmake_minimum_required(VERSION 3.10)

add_executable(main_minicpm_o main.cpp)
target_link_libraries(main_minicpm_o PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpm_o PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(main_minicpm_o2 mainllm.cpp)
target_link_libraries(main_minicpm_o2 PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpm_o2 PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(main_minicpm_dbg main_dbg.cpp)
target_link_libraries(main_minicpm_dbg PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpm_dbg PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE MllmRT MllmCPUBackend)
target_include_directories(tokenizer_test PRIVATE ${MLLM_INCLUDE_DIR})
