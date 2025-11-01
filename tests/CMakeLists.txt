add_subdirectory(nn)
add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(plugin)
add_subdirectory(engine)
add_subdirectory(compile)
add_subdirectory(async_io)

# Backends
add_subdirectory(cpu)
if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(cuda)
endif()
