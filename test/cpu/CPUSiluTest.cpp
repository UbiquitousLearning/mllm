//
// Created by Xiang Li on 23-10-17.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUSiLU.hpp"
TEST_F(CPUTest, CPUSilu1) {
    SETUP_OP(CPUSiLU, 4);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(c_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);
    TEST_RESHAPE({input0}, {c_output});
    TEST_SETUP({input0}, {c_output});
    PRINT_TENSOR_SHAPES(input0, c_output, output);
    TEST_EXCUTE({input0}, {c_output});
    // output->printData<float>();
    COMPARE_TENSOR(c_output, output, true);
}