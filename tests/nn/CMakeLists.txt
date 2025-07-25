if(MLLM_BUILD_ARM_BACKEND)
  set(MLLM_HOST_BACKEND_NN_TEST MllmArmBackend)
else()
  set(MLLM_HOST_BACKEND_NN_TEST MllmX86Backend)
endif()


add_executable(Mllm-Test-Nn-FooNetTest FooNetTest.cpp)
target_link_libraries(Mllm-Test-Nn-FooNetTest PRIVATE MllmRT ${MLLM_HOST_BACKEND_NN_TEST})
target_include_directories(Mllm-Test-Nn-FooNetTest PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
