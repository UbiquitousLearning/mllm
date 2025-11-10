file(GLOB_RECURSE MLLM_RT_CORE_SRC ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp)
file(GLOB_RECURSE MLLM_RT_UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)
file(GLOB_RECURSE MLLM_RT_ENGINE_SRC ${CMAKE_CURRENT_LIST_DIR}/engine/*.cpp)
file(GLOB_RECURSE MLLM_RT_NN_SRC ${CMAKE_CURRENT_LIST_DIR}/nn/*.cpp)
file(GLOB_RECURSE MLLM_RT_MODELS_SRC ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
file(GLOB_RECURSE MLLM_RT_COMPILE_SRC ${CMAKE_CURRENT_LIST_DIR}/compile/*.cpp)
file(GLOB_RECURSE MLLM_RT_PREPROCESSOR_SRC ${CMAKE_CURRENT_LIST_DIR}/preprocessor/*.cpp)
if(MLLM_BUILD_EXPERIMENTS)
  file(GLOB_RECURSE MLLM_RT_AUTO_TUNE_SRC ${CMAKE_CURRENT_LIST_DIR}/experiments/auto_tune/*.cpp)
endif()
file(GLOB WENET_AUDIO_SOURCES ${PROJECT_SOURCE_DIR}/third_party/wenet_audio/*)

add_library(
  MllmRT SHARED
  mllm.cpp
  backends/base/Backend.cpp
  backends/base/Allocator.cpp
  backends/base/PluginSystem.cpp
  ${MLLM_RT_CORE_SRC}
  ${MLLM_RT_UTILS_SRC}
  ${MLLM_RT_ENGINE_SRC}
  ${MLLM_RT_NN_SRC}
  ${MLLM_RT_PREPROCESSOR_SRC}
  ${MLLM_RT_MODELS_SRC}
  ${MLLM_RT_COMPILE_SRC}
  ${MLLM_RT_AUTO_TUNE_SRC}
  ${WENET_AUDIO_SOURCES}
)

target_include_directories(MllmRT PUBLIC ${MLLM_INCLUDE_DIR} ${MLLM_JSON_INCLUDE_DIR})
target_link_libraries(MllmRT PUBLIC fmt::fmt-header-only STDEXEC::stdexec xxHash::xxhash)

# Compiler stack need mlir
if(MLLM_ENABLE_MLLM_COMPILE_STACK)
  set(MLIR_DIR ${MLLM_COMPILE_STACK_TOOL_PATH}/lib/cmake/mlir)
  message(STATUS "Finding things at ${MLIR_DIR}")
  find_package(MLIR REQUIRED CONFIG)
  target_link_libraries(MllmRT PRIVATE
    MLIRPass
  )
  target_include_directories(MllmRT PRIVATE
    ${MLIR_INCLUDE_DIRS}
  )
endif()

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

# ONLY APPLE CAN DO !
# Processing OpenMP
if(MLLM_KERNEL_USE_THREADS AND MLLM_KERNEL_THREADS_VENDOR_OPENMP)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR APPLE)
    target_link_libraries(MllmRT PUBLIC ${OpenMP_omp_LIBRARY})
    target_compile_options(MllmRT PUBLIC ${OpenMP_CXX_FLAGS})
    # Apple should not use OpenMP
    message(ERROR "Apple platform should not use OpenMP. Pls set MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=ON")
  else()
    target_link_libraries(MllmRT PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_options(MllmRT PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(MllmRT PUBLIC ${OpenMP_CXX_INCLUDE_DIR})
  endif()
endif()

# Handle performance components
if(MLLM_PERFETTO_ENABLE)
  if (WIN32)
    # The perfetto library contains many symbols, so it needs the big object
    # format.
    target_compile_options(perfetto PRIVATE "/bigobj")
    # Disable legacy features in windows.h.
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    # On Windows we should link to WinSock2.
    target_link_libraries(MllmRT PUBLIC ws2_32)
  endif (WIN32)
  # Enable standards-compliant mode when using the Visual Studio compiler.
  if (MSVC)
    target_compile_options(MllmRT PRIVATE "/permissive-")
  endif (MSVC)
    target_link_libraries(MllmRT PUBLIC perfetto)
  if (ANDROID)
    target_link_libraries(MllmRT PUBLIC log)
  endif ()
endif()

if(MLLM_TRACY_ENABLE)
  add_subdirectory(tracy_perf)
  target_link_libraries(MllmRT PUBLIC MllmTracy)
endif()

# Host backend will be build by default
add_subdirectory(backends/cpu)

if(MLLM_BUILD_EXPERIMENTS)
  # FIXME: Add experiments sub project here.
endif()

# MLLM's FFI Extension
add_subdirectory(ffi)

if(MLLM_BUILD_OPENCL_BACKEND)
  add_subdirectory(backends/opencl)
endif()

if(MLLM_BUILD_QNN_BACKEND)
  add_subdirectory(backends/qnn)
  add_compile_definitions(
    MLLM_QNN_BACKEND
  )
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(backends/cuda)
  add_compile_definitions(
    MLLM_CUDA_BACKEND
  )
endif()

if(MLLM_BUILD_ASCEND_BACKEND)
  add_subdirectory(backends/ascend)
  add_compile_definitions(
    MLLM_ASCEND_BACKEND
  )
  set(MLLM_ASCEND_NPU_SOC "Ascend310P3" CACHE STRING "system on chip type")
  if(MLLM_ASCEND_CPU_DEBUG_MODE)
    add_compile_definitions(
      MLLM_ASCEND_CPU_DEBUG_MODE=1
    )
    set(MLLM_ASCEND_NPU_RUN_MODE "cpu" CACHE STRING "cpu/npu")
  else()
    set(MLLM_ASCEND_NPU_RUN_MODE "npu" CACHE STRING "cpu/npu")
  endif()
endif()

# Extension
if(MLLM_EXT_ENABLE)
  add_subdirectory(ext)
endif()
