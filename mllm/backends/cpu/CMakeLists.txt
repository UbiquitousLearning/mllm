# set(HWY_ENABLE_TESTS OFF)
# set(HWY_ENABLE_EXAMPLES OFF)
# set(HWY_ENABLE_CONTRIB OFF)
# set(BUILD_TESTING OFF)
# add_subdirectory(vendors/highway)

file(GLOB_RECURSE MLLM_CPU_BACKEND_OPS_SRC ${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp)
file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/*.cpp)

add_library(
  MllmCPUBackend SHARED
  ${MLLM_CPU_BACKEND_OPS_SRC} 
  ${MLLM_CPU_BACKEND_KERNELS_SRC} 
  CPUAllocator.cpp
  CPUBackend.cpp
  CPUDispatcher.cpp)

target_link_libraries(MllmCPUBackend PUBLIC MllmRT)
target_compile_options(MllmCPUBackend PUBLIC ${MLLM_CPU_BACKEND_COMPILE_OPTIONS})

install(
  TARGETS MllmCPUBackend
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
