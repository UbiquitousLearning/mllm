cmake_minimum_required(VERSION 3.10)

add_executable(main_image image_infer.cpp)
target_link_libraries(main_image PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_image PRIVATE ${MLLM_INCLUDE_DIR})

# Audio test executable
add_executable(main_audio audio_infer.cpp)
target_link_libraries(main_audio PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_audio PRIVATE ${MLLM_INCLUDE_DIR})

# End to end test executable
add_executable(main_minicpmo main.cpp)
target_link_libraries(main_minicpmo PRIVATE MllmRT MllmCPUBackend)
target_include_directories(main_minicpmo PRIVATE ${MLLM_INCLUDE_DIR})


