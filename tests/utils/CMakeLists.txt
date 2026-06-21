add_executable(Mllm-Test-Dbg DbgTest.cpp)
target_link_libraries(Mllm-Test-Dbg PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Dbg PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-Print PrintTest.cpp)
target_link_libraries(Mllm-Test-Print PRIVATE gtest_main MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Print PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
