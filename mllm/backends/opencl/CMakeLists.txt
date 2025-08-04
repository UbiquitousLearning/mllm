file(GLOB_RECURSE MLLM_OPENCL_BACKEND_OPS_SRC ${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp)
file(GLOB_RECURSE MLLM_OPENCL_BACKEND_RT_SRC ${CMAKE_CURRENT_LIST_DIR}/runtime/*.cpp)

add_library(
  MllmOpenCLBackend SHARED
  ${MLLM_OPENCL_BACKEND_OPS_SRC}
  ${MLLM_OPENCL_BACKEND_RT_SRC}
  OpenCLAllocator.cpp
  OpenCLBackend.cpp
  OpenCLDispatcher.cpp)

target_link_libraries(MllmOpenCLBackend PUBLIC MllmRT)
target_compile_options(MllmOpenCLBackend PUBLIC ${MLLM_OPENCL_BACKEND_COMPILE_OPTIONS})
target_include_directories(
  MllmOpenCLBackend
  PUBLIC
    ${MLLM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/vendors/OpenCL-Headers>
    $<INSTALL_INTERFACE:include/mllm/Backends/OpenCL/vendors/OpenCL-Headers/>)

add_library(MllmOpenCLBackend_ExtHeaders INTERFACE)
target_include_directories(
  MllmOpenCLBackend_ExtHeaders
  INTERFACE
    ${MLLM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/vendors/OpenCL-Headers>
    $<INSTALL_INTERFACE:include/mllm/Backends/OpenCL/vendors/OpenCL-Headers/>)

target_link_libraries(MllmOpenCLBackend INTERFACE MllmOpenCLBackend_ExtHeaders)

# ONLY APPLE CAN DO !
# Processing OpenMP
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR APPLE)
  target_link_libraries(MllmOpenCLBackend PUBLIC ${OpenMP_omp_LIBRARY})
  target_compile_options(MllmOpenCLBackend PUBLIC ${OpenMP_CXX_FLAGS})
else()
  target_link_libraries(MllmOpenCLBackend PUBLIC -fopenmp)
  target_compile_options(MllmOpenCLBackend PUBLIC -fopenmp)
endif()

install(
  TARGETS MllmOpenCLBackend MllmOpenCLBackend_ExtHeaders
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/vendors/OpenCL-Headers
  DESTINATION include/mllm/Backends/OpenCL/vendors
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")
