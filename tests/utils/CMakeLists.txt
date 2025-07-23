if(MLLM_BUILD_ARM_BACKEND)
  set(MLLM_HOST_BACKEND_UTILS_TEST MllmArmBackend)
else()
  set(MLLM_HOST_BACKEND_UTILS_TEST MllmX86Backend)
endif()


add_executable(Mllm-Test-Dbg DbgTest.cpp)
target_link_libraries(Mllm-Test-Dbg PRIVATE gtest_main MllmRT ${MLLM_HOST_BACKEND_UTILS_TEST})
target_include_directories(Mllm-Test-Dbg PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Print PrintTest.cpp)
target_link_libraries(Mllm-Test-Print PRIVATE gtest_main MllmRT ${MLLM_HOST_BACKEND_UTILS_TEST})
target_include_directories(Mllm-Test-Print PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
