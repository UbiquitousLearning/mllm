//
// Created by Xiang Li on 23-10-16.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUSoftMax.hpp"
TEST_F(CPUTest, CPUSoftMax1) {
    SETUP_OP(CPUSoftMax, DIMENSION, false, 4);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(c_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);
    TEST_RESHAPE({input0}, {c_output});
    TEST_SETUP({input0}, {c_output});
    PRINT_TENSOR_SHAPES(input0, output, c_output);
    TEST_EXCUTE({input0}, {c_output});
    COMPARE_TENSOR(output, c_output, true);
}