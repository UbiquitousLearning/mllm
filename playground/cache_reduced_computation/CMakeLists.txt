cmake_minimum_required(VERSION 3.21)
project(
  cache_reduced_computation
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

add_executable(cache_reduced_computation main.cpp)
target_link_libraries(cache_reduced_computation PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
target_include_directories(cache_reduced_computation PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)
