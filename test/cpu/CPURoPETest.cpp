//
// Created by lx on 23-10-17.
//

#include "CPUTest.hpp"
#include "backends/cpu/CPURoPE.hpp"
TEST_F(CPUTest, CPURoPE1) {
    GTEST_SKIP();
    SETUP_OP(CPURoPE, false, false);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(c_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);
    TEST_RESHAPE({input0}, {c_output});
    TEST_SETUP({input0}, {c_output});
    TEST_EXCUTE({input0}, {c_output});
    PRINT_TENSOR_SHAPES(input0, c_output, output);
    COMPARE_TENSOR(output, c_output);
}