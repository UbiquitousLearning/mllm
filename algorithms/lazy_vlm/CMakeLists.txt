cmake_minimum_required(VERSION 3.21)
project(
  lazy_vlm
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

file(GLOB_RECURSE MLLM_LAZY_VLM_MODELS ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)

add_executable(lazy_vlm_qwen2_5vl LazyVLMQwen2_5VL.cpp HKVCache.cpp ${MLLM_ALGORITHM_FANCY_MODELS})
target_link_libraries(lazy_vlm_qwen2_5vl PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
target_include_directories(lazy_vlm_qwen2_5vl PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)

add_executable(lazy_vlm_qwen2_5vl_fast LazyVLMQwen2_5VLFast.cpp HKVCacheFast.cpp ${MLLM_ALGORITHM_FANCY_MODELS})
target_link_libraries(lazy_vlm_qwen2_5vl_fast PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
target_include_directories(lazy_vlm_qwen2_5vl_fast PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)

add_executable(lazy_vlm_qwen2vl LazyVLMQwen2VL.cpp HKVCache.cpp ${MLLM_ALGORITHM_FANCY_MODELS})
target_link_libraries(lazy_vlm_qwen2vl PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
target_include_directories(lazy_vlm_qwen2vl PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)

add_executable(lazy_vlm_qwen2vl_fast LazyVLMQwen2VLFast.cpp HKVCacheFast.cpp ${MLLM_ALGORITHM_FANCY_MODELS})
target_link_libraries(lazy_vlm_qwen2vl_fast PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
target_include_directories(lazy_vlm_qwen2vl_fast PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)
