cmake_minimum_required(VERSION 3.10)

add_executable(main_minicpm_o main.cpp)
target_link_libraries(main_minicpm_o PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpm_o PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(main_minicpm_o_test maintest.cpp)
target_link_libraries(main_minicpm_o_test PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpm_o_test PRIVATE ${MLLM_INCLUDE_DIR})
