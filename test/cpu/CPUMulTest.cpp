//
// Created by Xiang Li on 23-10-16.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUMul.hpp"
TEST_F(CPUTest, CPUMul1) {
    SETUP_OP(CPUMul, 4);
    TENSOR(input0);
    TENSOR(input1);
    TENSOR(output);
    TENSOR(c_output);
    TEST_LOAD(input0);
    TEST_LOAD(input1);
    TEST_LOAD(output);
    TEST_RESHAPE({input0, input1}, {c_output});
    TEST_SETUP({input0, input1}, {c_output});
    TEST_EXCUTE({input0, input1}, {c_output});
    PRINT_TENSOR_SHAPES(input1, input0, output, c_output);
    COMPARE_TENSOR(c_output.get(), output.get(), true);
}