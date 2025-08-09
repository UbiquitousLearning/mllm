# if(MLLM_BUILD_ARM_BACKEND)
#   add_executable(Mllm-Benchmark-ARM-HPC-Sgemm arm_hpc_sgemm.cpp)
#   target_link_libraries(Mllm-Benchmark-ARM-HPC-Sgemm PRIVATE benchmark::benchmark MllmRT MllmCPUBackend)
# endif()