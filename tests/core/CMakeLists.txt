if(MLLM_BUILD_ARM_BACKEND)
  set(MLLM_HOST_BACKEND_UTILS_TEST MllmArmBackend)
else()
  set(MLLM_HOST_BACKEND_UTILS_TEST MllmX86Backend)
endif()


add_executable(Mllm-Test-Core-LoadParams LoadParamsTest.cpp)
target_link_libraries(Mllm-Test-Core-LoadParams PRIVATE MllmRT ${MLLM_HOST_BACKEND_UTILS_TEST})
target_include_directories(Mllm-Test-Core-LoadParams PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
