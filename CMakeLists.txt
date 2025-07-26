cmake_minimum_required(VERSION 3.21)
project(
  mllm
  VERSION 2.0.0
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MLLM_ENABLE_TEST "Enable MLLM test" ON)
option(MLLM_ENABLE_BENCHMARK "Enable MLLM benchmark" ON)
option(MLLM_ENABLE_EXAMPLE "Enable MLLM example" ON)
option(MLLM_ENABLE_TOOLS "Enable MLLM tools" ON)
option(MLLM_ENABLE_PY_MLLM "Enable MLLM python binding" OFF)
option(MLLM_ENABLE_JIT "Enable MLLM jit" OFF)
option(MLLM_BUILD_ARM_BACKEND "Enable MLLM ARM backend" OFF)
option(MLLM_CROSS_COMPILE "If in MLLM Cross Compile Mode. If not X86 backends will be built by default" OFF)
option(MLLM_BUILD_OPENCL_BACKEND "Enable MLLM OpenCL backend" OFF)
option(MLLM_BUILD_CUDA_BACKEND "Enable MLLM CUDA backend" OFF)
option(MLLM_BUILD_QNN_BACKEND "Enable MLLM QNN backend" OFF)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

include(cmake/CPM.cmake)
set(STDEXEC_BUILD_EXAMPLES OFF)
CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG nvhpc-25.03.rc1
)

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
    SHARED
    ${PROJECT_SOURCE_DIR}/pymllm/_C/PyWrap.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Nn.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Core.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Engine.cpp)
  target_compile_options(_C PUBLIC ${_py_compile_opts})
  target_link_libraries(_C PRIVATE ${_py_dep_libs})

  install(TARGETS _C MllmRT MllmCPUBackend LIBRARY DESTINATION pymllm/)
endif()
