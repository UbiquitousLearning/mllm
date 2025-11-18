file(GLOB MLLM_ASCEND_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

message(STATUS "SOC_VERSION is ${SOC_VERSION}, RUN_MODE is ${RUN_MODE}")
if("${RUN_MODE}" STREQUAL "cpu")
  include(cmake/cpu_lib.cmake)
else()
  include(cmake/npu_lib.cmake)
endif()

# ============ The CATLASS Code ============
# TODO add catlass

# ============ MLLM Code ============
add_library(
  MllmAscendBackend SHARED
  AscendAllocator.cpp
  AscendBackend.cpp
  AscendCommon.cpp
  AscendDispatcher.cpp
)
target_link_libraries(MllmAscendBackend PRIVATE
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
  ascendcl
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:MllmAscendKernel>>
  MllmRT
)
target_compile_definitions(MllmAscendBackend PUBLIC ASCENDC_DUMP=0)
set_target_properties(MllmAscendBackend PROPERTIES CXX_STANDARD 20)
target_compile_options(MllmAscendBackend PUBLIC 
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>> 
  -O2
)
target_include_directories(MllmAscendBackend PUBLIC "${MLLM_ASCEND_INSTALL_PATH}/include/")
