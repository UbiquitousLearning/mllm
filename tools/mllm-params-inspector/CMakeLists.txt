if(MLLM_BUILD_ARM_BACKEND)
  set(MLLM_HOST_BACKEND_TOOLS MllmArmBackend)
else()
  set(MLLM_HOST_BACKEND_TOOLS MllmX86Backend)
endif()


add_executable(mllm-params-inspector main.cpp)
target_link_libraries(mllm-params-inspector PRIVATE MllmRT ${MLLM_HOST_BACKEND_TOOLS})
target_include_directories(mllm-params-inspector PRIVATE ${MLLM_INCLUDE_DIR})
