add_executable(Mllm-Test-Dbg DbgTest.cpp)
target_link_libraries(Mllm-Test-Dbg PRIVATE gtest_main MllmRT)
target_include_directories(Mllm-Test-Dbg PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
