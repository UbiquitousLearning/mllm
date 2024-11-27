//
// Created by Xiang Li on 23-10-15.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPURMSNorm.hpp"
TEST_F(CPUTest, CPURMSNorm1) {
    SETUP_OP(CPURMSNorm, 32000, 1e-5, false);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(c_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {c_output});
    TEST_SETUP({input0}, {c_output});
    // TEST_LOAD(&op->weight(), false);
    TEST_WEIGHTS_LOAD(loader);
    //    op->weight().printData<float>();
    TEST_EXCUTE({input0}, {c_output});
    COMPARE_TENSOR(c_output.get(), output.get(), true);
}