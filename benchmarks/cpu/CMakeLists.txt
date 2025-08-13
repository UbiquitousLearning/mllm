if(MLLM_BUILD_ARM_BACKEND)
  add_executable(Mllm-Benchmark-ARM-HPC-Sgemm arm_mllm_blas_sgemm.cpp)
  target_link_libraries(Mllm-Benchmark-ARM-HPC-Sgemm PRIVATE benchmark::benchmark MllmRT MllmCPUBackend)
endif()