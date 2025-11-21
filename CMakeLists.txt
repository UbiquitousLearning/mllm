cmake_minimum_required(VERSION 3.21)
project(
  mllm
  VERSION 2.0.0
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MLLM Framework related
option(MLLM_ENABLE_TEST "Enable MLLM test" ON)
option(MLLM_ENABLE_BENCHMARK "Enable MLLM benchmark" ON)
option(MLLM_ENABLE_EXAMPLE "Enable MLLM example" ON)
option(MLLM_ENABLE_TOOLS "Enable MLLM tools" ON)
option(MLLM_ENABLE_PY_MLLM "Enable MLLM python binding" OFF)
option(MLLM_ENABLE_JIT "Enable MLLM jit" OFF)
option(MLLM_BUILD_ARM_BACKEND "Enable MLLM ARM backend" OFF)
option(MLLM_BUILD_OPENCL_BACKEND "Enable MLLM OpenCL backend" OFF)
option(MLLM_BUILD_CUDA_BACKEND "Enable MLLM CUDA backend" OFF)
option(MLLM_BUILD_QNN_BACKEND "Enable MLLM QNN backend" OFF)
option(MLLM_BUILD_ASCEND_BACKEND "Enable MLLM Ascend backend" OFF)
option(MLLM_BUILD_SDK_C_BINDING "Enable MLLM C SDK binding" ON)
option(MLLM_BUILD_EXPERIMENTS "Enable MLLM experiments" OFF)
option(MLLM_BUILD_EXT_OP_SET "Enable building mllm extension op set." ON)
option(MLLM_BUILD_EXT_OP_SET_TEST "Enable building mllm extension op set test." ON)

# MLLM Compile Stack
option(MLLM_ENABLE_MLLM_COMPILE_STACK "Enable compile stack of mllm. You need prebuilt mlir first." OFF)

# Extension Enable
option(MLLM_EXT_ENABLE OFF)
option(MLLM_EXT_ENABLE_LLVM_PROJECT OFF)
option(MLLM_EXT_ENABLE_META_TORCH_TOKENIZERS OFF)

# CPU Backend: BLAS
option(MLLM_USE_BLAS "Enable BLAS" OFF)
option(MLLM_BLAS_VENDOR_ACCELERATE "Enable Accelerate BLAS on OSX" OFF)
option(MLLM_BLAS_VENDOR_MKL "Enable MKL BLAS on X86" OFF)
option(MLLM_BLAS_VENDOR_BLIS "Enable BLIS BLAS for multi-platform" OFF)

# CPU Backend: SME2 and SVE2
option(MLLM_CPU_BACKEND_USE_SME2 "Enable SME2" OFF)

# Ascend Backend: Options
option(MLLM_ASCEND_CPU_DEBUG_MODE "Enable CPU Debug mode in ascend" OFF)

# Threads
option(MLLM_KERNEL_USE_THREADS "Enable Threads" ON)
option(MLLM_KERNEL_USE_THREADS_VENDOR_MLLM "Enable mllm's thread pool" ON)
option(MLLM_KERNEL_THREADS_VENDOR_OPENMP "Enable OpenMP Threads" OFF)
option(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD "Enable Apple GCD Threads" OFF)

# Performance profile components
option(MLLM_PERFETTO_ENABLE "Enable perfetto" OFF)
option(MLLM_TRACY_ENABLE "Enable Tracy. A more advanced profiler" OFF)

# Platform Hints
option(MLLM_ANDROID_BURST_PERFORMANCE_HINTS "If MLLM need use APerformanceHintManager to tell android we need best performance" OFF)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

# Get Git Commit
execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE MLLM_GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(NOT MLLM_GIT_COMMIT_HASH)
  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE MLLM_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if(NOT MLLM_GIT_COMMIT_HASH)
  set(MLLM_GIT_COMMIT_HASH "unknown")
endif()
message(STATUS "MLLM_GIT_COMMIT_HASH=${MLLM_GIT_COMMIT_HASH}")
add_compile_definitions(MLLM_GIT_COMMIT_HASH=${MLLM_GIT_COMMIT_HASH})

if(MLLM_ENABLE_PY_MLLM)
  add_compile_definitions(MLLM_ENABLE_PY_MLLM=1)
  if(APPLE)
    set(MLLM_BUILD_ARM_BACKEND ON CACHE BOOL "Build ARM backend" FORCE)
    set(MLLM_CPU_BACKEND_COMPILE_OPTIONS "-march=native+fp16+fp16fml+dotprod+i8mm+sme" CACHE STRING "CPU compile options" FORCE)
    set(MLLM_USE_BLAS ON CACHE BOOL "Use BLAS" FORCE)
    set(MLLM_BLAS_VENDOR_ACCELERATE ON CACHE BOOL "Use Apple Accelerate framework" FORCE)
    set(MLLM_KERNEL_USE_THREADS ON CACHE BOOL "Enable Threads" FORCE)
    set(MLLM_KERNEL_THREADS_VENDOR_OPENMP OFF CACHE BOOL "Disable OpenMP Threads" FORCE)
    set(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD ON CACHE BOOL "Enable Apple GCD Threads" FORCE)
    set(MLLM_BUILD_EXPERIMENTS ON CACHE BOOL "Build experimental features" FORCE)
  else() # Other system. Linux or windows that run on X86\RISC, etc. (ARM is not considered in this scope.)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable HWY tests" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable HWY examples" FORCE)
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable HWY contrib" FORCE)
    set(MLLM_CPU_BACKEND_COMPILE_OPTIONS "-march=native" CACHE STRING "CPU compile options for x86" FORCE)
    set(MLLM_KERNEL_USE_THREADS ON CACHE BOOL "Enable Threads" FORCE)
    set(MLLM_KERNEL_THREADS_VENDOR_OPENMP ON CACHE BOOL "Enable OpenMP Threads" FORCE)
    set(MLLM_BUILD_EXPERIMENTS ON CACHE BOOL "Build experimental features" FORCE)
  endif()
endif()

include(cmake/CPM.cmake)
set(STDEXEC_BUILD_EXAMPLES OFF)
CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG nvhpc-25.03.rc1
)

# ONLY APPLE CAN DO !
# Apple clang's driver removes the -fopenmp flag, so we need to add it manually.
if(MLLM_KERNEL_USE_THREADS)
  if(MLLM_KERNEL_USE_THREADS_VENDOR_MLLM)
    add_compile_definitions(MLLM_KERNEL_USE_THREADS_VENDOR_MLLM=1)
  endif()
  # Compile Option
  add_compile_definitions(MLLM_KERNEL_USE_THREADS=1)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR APPLE)
    # Check if both OpenMP and GCD are enabled
    if(MLLM_KERNEL_THREADS_VENDOR_OPENMP AND MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD)
      message(WARNING "Both OpenMP and GCD are enabled on Apple platform. Using GCD only, OpenMP flags will be empty.")
      # Use GCD only, set OpenMP flags to empty
      set(OpenMP_C_FLAGS "")
      set(OpenMP_CXX_FLAGS "")
      set(OpenMP_C_LIB_NAMES "")
      set(OpenMP_CXX_LIB_NAMES "")
      set(OpenMP_omp_LIBRARY "")
      # Compile Option
      add_compile_definitions(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=1)
    elseif(MLLM_KERNEL_THREADS_VENDOR_OPENMP)
      message(WARNING "OpenMP is enabled on Apple platform. Consider using GCD (MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=ON) for better performance on Apple systems.")
      execute_process(COMMAND brew --prefix libomp
              OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
              OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(OpenMP_C_FLAGS "-Xpreprocessor" "-fopenmp" "-I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor" "-fopenmp" "-I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES omp)
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      find_package(OpenMP REQUIRED)
      # Compile Option
      add_compile_definitions(MLLM_KERNEL_THREADS_VENDOR_OPENMP=1)
    elseif(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD)
      # GCD only, no OpenMP
      set(OpenMP_C_FLAGS "")
      set(OpenMP_CXX_FLAGS "")
      set(OpenMP_C_LIB_NAMES "")
      set(OpenMP_CXX_LIB_NAMES "")
      set(OpenMP_omp_LIBRARY "")
      # Compile Option
      add_compile_definitions(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=1)
    endif()
  else()
    # Non-Apple platforms
    if(MLLM_KERNEL_THREADS_VENDOR_OPENMP)
      if(ANDROID)
        file(GLOB_RECURSE _omp_h "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/*/lib/clang/*/include/omp.h")
        if(_omp_h)
            get_filename_component(OpenMP_CXX_INCLUDE_DIR "${_omp_h}" DIRECTORY)
        endif()
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_C_FLAGS "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "")
      else()
        find_package(OpenMP)
        if(NOT OpenMP_FOUND)
          # Some platforms can use -fopenmp directly even if find_package fails
          set(OpenMP_CXX_FLAGS "-fopenmp")
          set(OpenMP_C_FLAGS "-fopenmp")
          message(STATUS "OpenMP not found via find_package, but trying with -fopenmp flag directly")
        endif()
      endif()
      # Compile Option
      add_compile_definitions(MLLM_KERNEL_THREADS_VENDOR_OPENMP=1)
      message(STATUS "Using OpenMP via find_package")
      message(STATUS "OpenMP CXX_FLAGS=${OpenMP_CXX_FLAGS}")
      message(STATUS "OpenMP CXX_LIB_NAMES=${OpenMP_CXX_LIB_NAMES}")
      message(STATUS "OpenMP OpenMP_omp_LIBRARY=${OpenMP_omp_LIBRARY}")
      message(STATUS "OpenMP OpenMP_CXX_INCLUDE_DIR=${OpenMP_CXX_INCLUDE_DIR}")
    else()
      # No threading or other threading implementation
      set(OpenMP_C_FLAGS "")
      set(OpenMP_CXX_FLAGS "")
      set(OpenMP_C_LIB_NAMES "")
      set(OpenMP_CXX_LIB_NAMES "")
      set(OpenMP_omp_LIBRARY "")
    endif()
  endif()
else()
  set(OpenMP_C_FLAGS "")
  set(OpenMP_CXX_FLAGS "")
  set(OpenMP_C_LIB_NAMES "")
  set(OpenMP_CXX_LIB_NAMES "")
  set(OpenMP_omp_LIBRARY "")
endif()

# Handle Performance components
if(MLLM_PERFETTO_ENABLE)
  include_directories(third_party/perfetto/sdk)
  add_library(perfetto STATIC third_party/perfetto/sdk/perfetto.cc)
  add_compile_definitions(MLLM_PERFETTO_ENABLE=1)
endif()

if(MLLM_TRACY_ENABLE)
  # Fetch Tracy
  FetchContent_Declare (
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.12.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(tracy)
  add_compile_definitions(MLLM_TRACY_ENABLE=1)
endif()

# Handle Windows Symbols.
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Set binary place
if(NOT MLLM_ENABLE_PY_MLLM)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Setup for installation.
set(MLLM_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxHash/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/utfcpp/include>
    $<INSTALL_INTERFACE:include/mllm>
    $<INSTALL_INTERFACE:include/third_party/>)
set(MLLM_JSON_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/json>
    $<INSTALL_INTERFACE:include/third_party/>)

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/xxHash)
add_subdirectory(mllm)

if(MLLM_ENABLE_TEST)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()

if(MLLM_ENABLE_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  add_subdirectory(benchmarks)
endif()

if(MLLM_ENABLE_EXAMPLE)
  add_subdirectory(examples)
endif()

if(MLLM_ENABLE_TOOLS)
  add_subdirectory(tools)
endif()

if(MLLM_BUILD_EXT_OP_SET)
  add_subdirectory(mllm-ext-opset)
endif()

# Handle SDK C Bindings
if (MLLM_BUILD_SDK_C_BINDING)
  file(GLOB_RECURSE MLLM_C_API_SRC ${CMAKE_CURRENT_LIST_DIR}/mllm/c_api/*.cpp)
  add_library(MllmSdkC SHARED ${MLLM_C_API_SRC})
  target_link_libraries(MllmSdkC PRIVATE MllmRT MllmCPUBackend)
endif()

# Install mllm main library: Config.
set(INSTALL_MLLM_VERSION ${PROJECT_VERSION})
set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "Installation directory for headers")

if(MLLM_PERFETTO_ENABLE)
  install(
    TARGETS perfetto
    EXPORT MllmTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/perfetto/sdk/
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "Vendors/*" EXCLUDE)
endif()

install(
  TARGETS MllmRT
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  TARGETS xxhash_static
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

if(MLLM_BUILD_SDK_C_BINDING)
  install(
    TARGETS MllmSdkC
    EXPORT MllmTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
endif()

if(MLLM_TRACY_ENABLE)
  install(
    TARGETS MllmTracy
    EXPORT MllmTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm/
  DESTINATION include/mllm
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.inl"
  PATTERN "Vendors/*" EXCLUDE)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm-ext-opset/
  DESTINATION include/mllm-ext-opset
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.inl"
  PATTERN "Vendors/*" EXCLUDE)

# Embedding fmt lib into mllm packages to avoid fetching fmt.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt
        DESTINATION packages)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/include/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/include/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include/dlpack/
  DESTINATION include/dlpack/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utfcpp/include/utfcpp/
  DESTINATION include/utfcpp/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxHash/include/xxHash/
  DESTINATION include/xxHash/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

# Install tools
# FIXME: Add other tools here
install(TARGETS mllm-params-inspector mllm-quantizer DESTINATION bin)

# Final config and file copy.
install(
  EXPORT MllmTargets
  FILE MllmTargets.cmake
  NAMESPACE mllm::
  DESTINATION lib/cmake/)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake
  DESTINATION lib/cmake/)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mllmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mllmConfig.cmake
  INSTALL_DESTINATION lib/cmake/
  PATH_VARS INCLUDE_INSTALL_DIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mllmConfigVersion.cmake
  VERSION ${INSTALL_MLLM_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mllmConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/mllmConfigVersion.cmake
        DESTINATION lib/cmake/)
