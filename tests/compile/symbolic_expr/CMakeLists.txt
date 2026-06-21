add_executable(Mllm-Test-SymbolicExpr SymbolicExprTest.cpp)
target_link_libraries(Mllm-Test-SymbolicExpr PRIVATE gtest_main MllmCPUBackend MllmRT)
target_include_directories(Mllm-Test-SymbolicExpr PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
