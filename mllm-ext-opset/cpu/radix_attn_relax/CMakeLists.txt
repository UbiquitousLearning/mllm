add_library(MllmExtOpSet_CPU_RadixAttnRelax SHARED RadixAttnRelax.cpp)
target_link_libraries(MllmExtOpSet_CPU_RadixAttnRelax PRIVATE MllmRT MllmCPUBackend)
target_include_directories(MllmExtOpSet_CPU_RadixAttnRelax PRIVATE ${MLLM_INCLUDE_DIR})

install(
  TARGETS MllmExtOpSet_CPU_RadixAttnRelax
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

# if(MLLM_BUILD_EXT_OP_SET_TEST)
#   add_executable(Mllm-Test-ExtOpSet-CPU-RadixAttnRelax tests/main.cpp)
#   target_link_libraries(Mllm-Test-ExtOpSet-CPU-RadixAttnRelax PRIVATE gtest_main MllmRT MllmCPUBackend)
#   target_include_directories(Mllm-Test-ExtOpSet-CPU-RadixAttnRelax PRIVATE ${MLLM_INCLUDE_DIR})
#   include(GoogleTest)
# endif()
