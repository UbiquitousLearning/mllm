cmake_minimum_required(VERSION 3.21)
project(
  fancy_algorithm
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

file(GLOB_RECURSE MLLM_ALGORITHM_FANCY_MODELS
     ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
file(GLOB_RECURSE MLLM_ALGORITHM_FANCY_PASSES
     ${CMAKE_CURRENT_LIST_DIR}/passes/*.cpp)

add_executable(fancy_algorithm main.cpp ${MLLM_ALGORITHM_FANCY_MODELS} ${MLLM_ALGORITHM_FANCY_PASSES})
target_link_libraries(fancy_algorithm PRIVATE mllm::MllmRT)
target_include_directories(fancy_algorithm PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../)
