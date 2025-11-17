cmake_minimum_required(VERSION 3.10)

# add_executable(main_minicpm_o main.cpp)
# target_link_libraries(main_minicpm_o PRIVATE MllmRT MllmCPUBackend)
# target_include_directories(main_minicpm_o PRIVATE ${MLLM_INCLUDE_DIR})

# add_executable(main_minicpm_o_test maintest.cpp)
# target_link_libraries(main_minicpm_o_test PRIVATE MllmRT MllmCPUBackend)
# target_include_directories(main_minicpm_o_test PRIVATE ${MLLM_INCLUDE_DIR})

# Audio test executable
add_executable(main_audio_test main_audio_test.cpp)
target_link_libraries(main_audio_test PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_audio_test PRIVATE ${MLLM_INCLUDE_DIR})

#add_executable(stream-demo demo_minicpmo_streaming.cpp)
#target_link_libraries(stream-demo PRIVATE MllmRT MllmCPUBackend)
#target_include_directories(stream-demo PRIVATE ${MLLM_INCLUDE_DIR})

