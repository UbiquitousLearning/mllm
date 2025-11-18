add_library(MllmExtOpSet_CPU_FlashAttn2SwaSink SHARED FlashAttn2SwaSink.cpp)
target_link_libraries(MllmExtOpSet_CPU_FlashAttn2SwaSink PRIVATE MllmRT MllmCPUBackend)
target_include_directories(MllmExtOpSet_CPU_FlashAttn2SwaSink PRIVATE ${MLLM_INCLUDE_DIR})

install(
  TARGETS MllmExtOpSet_CPU_FlashAttn2SwaSink
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

if(MLLM_BUILD_EXT_OP_SET_TEST)
  add_executable(Mllm-Test-ExtOpSet-CPU-FlashAttn2SwaSink tests/main.cpp)
  target_link_libraries(Mllm-Test-ExtOpSet-CPU-FlashAttn2SwaSink PRIVATE gtest_main MllmRT MllmCPUBackend)
  target_include_directories(Mllm-Test-ExtOpSet-CPU-FlashAttn2SwaSink PRIVATE ${MLLM_INCLUDE_DIR})
  include(GoogleTest)
endif()
