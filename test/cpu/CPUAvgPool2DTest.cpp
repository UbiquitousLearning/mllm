//
// Created by Xiang Li on 23-10-15.
//
#include "backends/cpu/op/CPUAvgPool2D.hpp"

#include "CPUTest.hpp"
TEST_F(CPUTest, CPUAvgPool2D1) {
    SETUP_OP(CPUAvgPool2D, {3, 3}, {1, 1}, VALID, 4);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    PRINT_TENSOR_SHAPES(input0, output, test_output);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get(), true);
}

TEST_F(CPUTest, CPUAvgPool2D2) {
    SETUP_OP(CPUAvgPool2D, {3, 3}, {1, 1}, SAME, 4);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    PRINT_TENSOR_SHAPES(input0, output, test_output);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get(), true);
}