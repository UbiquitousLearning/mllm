file(GLOB_RECURSE MLLM_RT_CORE_SRC ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp)
file(GLOB_RECURSE MLLM_RT_UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)
file(GLOB_RECURSE MLLM_RT_ENGINE_SRC ${CMAKE_CURRENT_LIST_DIR}/engine/*.cpp)
file(GLOB_RECURSE MLLM_RT_NN_SRC ${CMAKE_CURRENT_LIST_DIR}/nn/*.cpp)
file(GLOB_RECURSE MLLM_RT_MODELS_SRC ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
file(GLOB_RECURSE MLLM_RT_COMPILE_SRC ${CMAKE_CURRENT_LIST_DIR}/compile/*.cpp)
file(GLOB_RECURSE MLLM_RT_PREPROCESSOR_SRC ${CMAKE_CURRENT_LIST_DIR}/preprocessor/*.cpp)
file(GLOB_RECURSE MLLM_RT_AUTO_TUNE_SRC ${CMAKE_CURRENT_LIST_DIR}/auto_tune/*.cpp)

add_library(
  MllmRT SHARED
  mllm.cpp
  backends/base/Backend.cpp
  backends/base/Allocator.cpp
  ${MLLM_RT_CORE_SRC}
  ${MLLM_RT_UTILS_SRC}
  ${MLLM_RT_ENGINE_SRC}
  ${MLLM_RT_NN_SRC}
  ${MLLM_RT_PREPROCESSOR_SRC}
  ${MLLM_RT_MODELS_SRC}
  ${MLLM_RT_COMPILE_SRC}
  ${MLLM_RT_AUTO_TUNE_SRC})

target_include_directories(MllmRT PUBLIC ${MLLM_INCLUDE_DIR} ${MLLM_JSON_INCLUDE_DIR})
target_link_libraries(MllmRT PUBLIC fmt::fmt-header-only STDEXEC::stdexec)

# stdexec will use Pack indexing feature. 
#
# Some compilers turn on pack indexing in pre-C++26 code. We want to use it if it is
# available. Pack indexing is disabled for clang < 20 because of:
# https://github.com/llvm/llvm-project/issues/116105
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  string(REGEX MATCH "^[0-9]+" COMPILER_VERSION_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
  if(COMPILER_VERSION_MAJOR LESS 20)
    target_compile_options(MllmRT PUBLIC -Wno-c++26-extensions)
  endif()
endif()

# Host backend will be build by default
add_subdirectory(backends/cpu)

if(MLLM_BUILD_OPENCL_BACKEND)
  add_subdirectory(backends/opencl)
endif()

if(MLLM_BUILD_QNN_BACKEND)
  add_subdirectory(backends/qnn)
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(backends/cuda)
endif()
