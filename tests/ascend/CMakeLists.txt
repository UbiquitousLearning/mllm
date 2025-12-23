add_executable(Mllm-Test-AscendKernel KernelTest.cpp)

if(DEFINED ENV{ASCEND_HOME_PATH})
  target_include_directories(Mllm-Test-AscendKernel PRIVATE $ENV{ASCEND_HOME_PATH}/include)
  target_link_directories(Mllm-Test-AscendKernel PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
endif()

target_link_libraries(Mllm-Test-AscendKernel PRIVATE
  gtest_main
  MllmRT
  MllmCPUBackend
  MllmAscendBackend
)

target_include_directories(Mllm-Test-AscendKernel PRIVATE
  ${CMAKE_SOURCE_DIR}
)

set_target_properties(Mllm-Test-AscendKernel PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

include(GoogleTest)

