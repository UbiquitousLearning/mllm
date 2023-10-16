//
// Created by lx on 23-10-15.
//
#include "CPUTest.hpp"
#include "backends/cpu/CPULinear.hpp"
TEST_F(CPUTest, CPULinear1) {
    SETUP_OP(CPULinear, 3, 4, true, false);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    TEST_LOAD(&op->weight_);
    TEST_LOAD(&op->bias_);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get());
}
TEST_F(CPUTest, CPULinear2) {
    SETUP_OP(CPULinear, 3, 4, false, false);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    TEST_LOAD(&op->weight_);
    //    TEST_LOAD(&op->bias_);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get());
}
TEST_F(CPUTest, CPULinear3) {
    SETUP_OP(CPULinear, 3, 4, false, false);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);
    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    TEST_LOAD(&op->weight_);
    //    TEST_LOAD(&op->bias_);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get());
}