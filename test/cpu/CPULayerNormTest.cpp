//
// Created by 咸的鱼 on 2023/11/27.
//

#include "CPUTest.hpp"
#include "backends/cpu/CPULayerNorm.hpp"
TEST_F(CPUTest, CPULayerNorm1) {
    SETUP_OP(CPULayerNorm, false);
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
    COMPARE_TENSOR(c_output.get(), output.get(), false);
}