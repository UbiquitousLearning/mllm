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

# CPU Backend: BLAS
option(MLLM_USE_BLAS "Enable BLAS" OFF)
option(MLLM_BLAS_VENDOR_ACCELERATE "Enable Accelerate BLAS on OSX" OFF)
option(MLLM_BLAS_VENDOR_MKL "Enable MKL BLAS on X86" OFF)
option(MLLM_BLAS_VENDOR_BLIS "Enable BLIS BLAS for multi-platform" OFF)

# Threads
option(MLLM_KERNEL_USE_THREADS "Enable Threads" ON)
option(MLLM_KERNEL_THREADS_VENDOR_OPENMP "Enable OpenMP Threads" ON)
option(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD "Enable Apple GCD Threads" OFF)

# Performance components
option(MLLM_PERFETTO_ENABLE "Enable perfetto" OFF)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

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
      find_package(OpenMP)
      if(NOT OpenMP_FOUND)
        # Some platforms can use -fopenmp directly even if find_package fails
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_C_FLAGS "-fopenmp")
        message(STATUS "OpenMP not found via find_package, but trying with -fopenmp flag directly")
      endif()
      # Compile Option
      add_compile_definitions(MLLM_KERNEL_THREADS_VENDOR_OPENMP=1)
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

# Handle NaN Warning.
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386|x86_64"))
  add_compile_options(-Wno-nan-infinity-disabled)
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
    $<INSTALL_INTERFACE:include/mllm>
    $<INSTALL_INTERFACE:include/third_party/>)
set(MLLM_JSON_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/json>
    $<INSTALL_INTERFACE:include/third_party/>)

add_subdirectory(third_party/fmt)
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
endif()

install(
  TARGETS MllmRT
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm/
  DESTINATION include/mllm
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
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
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

# Install tools
install(TARGETS mllm-params-inspector DESTINATION bin)

# Final config and file copy.
install(
  EXPORT MllmTargets
  FILE MllmTargets.cmake
  NAMESPACE mllm::
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

# Pymllm install
if(MLLM_ENABLE_PY_MLLM)
  target_compile_options(MllmRT PRIVATE -fPIC)

  find_package(Python3 COMPONENTS Interpreter Development)
  include_directories(${Python3_INCLUDE_DIRS})
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

  set(_py_dep_libs MllmRT MllmCPUBackend)

  # pybind compile options
  set(_py_compile_opts
      # Override depends on RTTI.
      # -frtti
      # -fexceptions
      -fPIC)

  # pybind portable lib _C
  pybind11_add_module(
    _C
    MODULE
    ${PROJECT_SOURCE_DIR}/pymllm/_C/PyWrap.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Nn.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Core.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Engine.cpp)
  target_compile_options(_C PUBLIC ${_py_compile_opts})
  target_link_libraries(_C PRIVATE ${_py_dep_libs})

  install(TARGETS _C MllmRT MllmCPUBackend LIBRARY DESTINATION pymllm/)
endif()
