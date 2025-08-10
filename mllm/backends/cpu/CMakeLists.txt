# set(HWY_ENABLE_TESTS OFF)
# set(HWY_ENABLE_EXAMPLES OFF)
# set(HWY_ENABLE_CONTRIB OFF)
# set(BUILD_TESTING OFF)
# add_subdirectory(vendors/highway)

file(GLOB_RECURSE MLLM_CPU_BACKEND_OPS_SRC ${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp)
file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_COMMON_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/common/*.cpp)

if(MLLM_BUILD_ARM_BACKEND)
  file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/*.cpp)
else()
  file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/x86/*.cpp)
endif()

add_library(
  MllmCPUBackend SHARED
  ${MLLM_CPU_BACKEND_OPS_SRC}
  ${MLLM_CPU_BACKEND_KERNELS_SRC}
  ${MLLM_CPU_BACKEND_KERNELS_COMMON_SRC}
  CPUAllocator.cpp
  CPUBackend.cpp
  CPUDispatcher.cpp)

target_link_libraries(MllmCPUBackend PUBLIC MllmRT)
target_compile_options(MllmCPUBackend PUBLIC ${MLLM_CPU_BACKEND_COMPILE_OPTIONS})

# ONLY APPLE CAN DO !
# Processing OpenMP
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR APPLE)
  target_link_libraries(MllmCPUBackend PUBLIC ${OpenMP_omp_LIBRARY})
  target_compile_options(MllmCPUBackend PUBLIC ${OpenMP_CXX_FLAGS})
else()
  target_link_libraries(MllmCPUBackend PUBLIC -fopenmp)
  target_compile_options(MllmCPUBackend PUBLIC -fopenmp)
endif()

if(MLLM_USE_BLAS)
  include(BLAS.cmake)
  mllm_configure_blas()
  if(MLLM_BLAS_LIBRARIES)
    target_link_libraries(MllmCPUBackend PUBLIC ${MLLM_BLAS_LIBRARIES})
  endif()
  if(MLLM_BLAS_COMPILE_DEFINITIONS)
    target_compile_definitions(MllmCPUBackend PUBLIC ${MLLM_BLAS_COMPILE_DEFINITIONS})
  endif()
  if(MLLM_BLAS_INCLUDE_DIRS)
    target_include_directories(MllmCPUBackend PUBLIC ${MLLM_BLAS_INCLUDE_DIRS})
  endif()
endif()

install(
  TARGETS MllmCPUBackend
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
