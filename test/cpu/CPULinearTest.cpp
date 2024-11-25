//
// Created by Xiang Li on 23-10-15.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPULinear.hpp"
TEST_F(CPUTest, CPULinear1) {
    SETUP_OP(CPULinear, 128, 128, true, 4);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    // TEST_LOAD(&op->weight());
    // TEST_LOAD(&op->bias());
    TEST_WEIGHTS_LOAD(loader);
    PRINT_TENSOR_SHAPES(input0, output, test_output);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get(), true);
}
TEST_F(CPUTest, CPULinear2) {
    SETUP_OP(CPULinear, 3, 4, false, 4);
    TENSOR(input0)
    TENSOR(output)
    TENSOR(test_output);
    TEST_LOAD(input0);
    TEST_LOAD(output);

    TEST_RESHAPE({input0}, {test_output});
    TEST_SETUP({input0}, {test_output});
    // TEST_LOAD(&op->weight());
    //        TEST_LOAD(&op->bias_());
    TEST_WEIGHTS_LOAD(loader);
    TEST_EXCUTE({input0}, {test_output});
    COMPARE_TENSOR(output.get(), test_output.get(), true);
}
// TEST_F(CPUTest, CPULinear3) {
//     SETUP_OP(CPULinear, 3, 4, false, 4);
//     TENSOR(input0)
//     TENSOR(output)
//     TENSOR(test_output);
//     TEST_LOAD(input0);
//     TEST_LOAD(output);
//     TEST_RESHAPE({input0}, {test_output});
//     TEST_SETUP({input0}, {test_output});
//     TEST_LOAD(&op->weight());
//     //    TEST_LOAD(&op->bias_());
//     TEST_EXCUTE({input0}, {test_output});
//     COMPARE_TENSOR(output.get(), test_output.get());
// }