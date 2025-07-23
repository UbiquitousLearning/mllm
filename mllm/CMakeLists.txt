file(GLOB_RECURSE MLLM_RT_CORE_SRC ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp)
file(GLOB_RECURSE MLLM_RT_UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)
file(GLOB_RECURSE MLLM_RT_ENGINE_SRC ${CMAKE_CURRENT_LIST_DIR}/engine/*.cpp)
file(GLOB_RECURSE MLLM_RT_NN_SRC ${CMAKE_CURRENT_LIST_DIR}/nn/*.cpp)
file(GLOB_RECURSE MLLM_RT_MODELS_SRC ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
file(GLOB_RECURSE MLLM_RT_COMPILE_SRC ${CMAKE_CURRENT_LIST_DIR}/compile/*.cpp)
file(GLOB_RECURSE MLLM_RT_PREPROCESSOR_SRC ${CMAKE_CURRENT_LIST_DIR}/preprocessor/*.cpp)

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
  ${MLLM_RT_COMPILE_SRC})

target_include_directories(MllmRT PUBLIC ${MLLM_INCLUDE_DIR}
                                         ${MLLM_JSON_INCLUDE_DIR})
target_link_libraries(MllmRT PUBLIC fmt::fmt-header-only)

if(MLLM_BUILD_ARM_BACKEND)
  add_subdirectory(backends/arm)
endif()

if(MLLM_BUILD_OPENCL_BACKEND)
  add_subdirectory(backends/opencl)
endif()

if(MLLM_BUILD_QNN_BACKEND)
  add_subdirectory(backends/qnn)
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(backends/cuda)
endif()

if(NOT MLLM_CROSS_COMPILE)
  add_subdirectory(backends/x86)
endif()
