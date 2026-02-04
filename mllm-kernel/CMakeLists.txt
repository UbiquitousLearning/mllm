cmake_minimum_required(VERSION 3.21)
project(
  mllm_kernel
  VERSION 1.0.0
  LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(MLLM_KERNEL_BUILD_CPU "Build CPU kernels with Highway SIMD" ON)
option(MLLM_KERNEL_BUILD_CUDA "Build CUDA kernels" OFF)
option(MLLM_KERNEL_BUILD_ASCEND "Build Ascend kernels" OFF)

# Include CPM for package management
include(cmake/CPM.cmake)

# ============================================================================
# Highway SIMD Library (for CPU kernels)
# ============================================================================
if(MLLM_KERNEL_BUILD_CPU)
  # Disable Highway tests and examples to speed up build
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples" FORCE)
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable Highway contrib" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Disable testing" FORCE)

  CPMAddPackage(
    NAME highway
    GITHUB_REPOSITORY google/highway
    GIT_TAG 1.3.0
    OPTIONS
      "HWY_ENABLE_TESTS OFF"
      "HWY_ENABLE_EXAMPLES OFF"
      "HWY_ENABLE_CONTRIB OFF"
  )

  if(highway_ADDED)
    message(STATUS "Highway SIMD library added from: ${highway_SOURCE_DIR}")
    
    # Create an interface library to expose Highway includes
    add_library(mllm_kernel_highway INTERFACE)
    target_link_libraries(mllm_kernel_highway INTERFACE hwy)
    target_include_directories(mllm_kernel_highway INTERFACE
      $<BUILD_INTERFACE:${highway_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>
    )
  endif()
endif()

# ============================================================================
# Installation
# ============================================================================

# Install Highway headers for JIT compilation
if(MLLM_KERNEL_BUILD_CPU AND highway_ADDED)
  # Install Highway headers
  install(
    DIRECTORY ${highway_SOURCE_DIR}/hwy
    DESTINATION include
    FILES_MATCHING 
    PATTERN "*.h"
    PATTERN "*.inc"
  )
  
  # Install Highway library
  install(
    TARGETS hwy
    EXPORT MllmKernelTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
  
  # Install the interface target
  install(
    TARGETS mllm_kernel_highway
    EXPORT MllmKernelTargets
  )
endif()

# Install CPU kernel includes
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm_kernel/cpu/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

# Install CPU kernel sources (for JIT compilation)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm_kernel/cpu/csrc/
  DESTINATION share/mllm_kernel/cpu/csrc
  FILES_MATCHING
  PATTERN "*.cpp"
  PATTERN "*.hpp"
  PATTERN "*.h"
)

# Export targets
install(
  EXPORT MllmKernelTargets
  FILE MllmKernelTargets.cmake
  NAMESPACE mllm_kernel::
  DESTINATION lib/cmake/mllm_kernel
)

# Create and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MllmKernelConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MllmKernelConfig.cmake
  INSTALL_DESTINATION lib/cmake/mllm_kernel
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MllmKernelConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MllmKernelConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MllmKernelConfigVersion.cmake
  DESTINATION lib/cmake/mllm_kernel
)

# Print summary
message(STATUS "")
message(STATUS "=== mllm-kernel Configuration ===")
message(STATUS "  Version:     ${PROJECT_VERSION}")
message(STATUS "  CPU kernels: ${MLLM_KERNEL_BUILD_CPU}")
message(STATUS "  CUDA kernels: ${MLLM_KERNEL_BUILD_CUDA}")
message(STATUS "  Ascend kernels: ${MLLM_KERNEL_BUILD_ASCEND}")
if(MLLM_KERNEL_BUILD_CPU AND highway_ADDED)
  message(STATUS "  Highway version: 1.3.0")
  message(STATUS "  Highway source: ${highway_SOURCE_DIR}")
endif()
message(STATUS "=================================")
message(STATUS "")
