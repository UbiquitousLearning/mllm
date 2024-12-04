//
// Created by Xiang Li on 23-10-16.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUMatmul.hpp"
// TEST_F(CPUTest, CPUMatmul1) {
//     SETUP_OP(CPUMatmul, false, false, 4);
//     TENSOR(input0);
//     TENSOR(input1);
//     TENSOR(output);
//     TENSOR(c_output);
//     TEST_LOAD(input0);
//     TEST_LOAD(input1);
//     TEST_LOAD(output);
//     TEST_RESHAPE({input0, input1}, {c_output});
//     TEST_SETUP({input0, input1}, {c_output});
//     PRINT_TENSOR_SHAPES(input0, input1, c_output, output);
//
//     TEST_EXCUTE({input0, input1}, {c_output});
//     //    c_output->printData<float>();
//     COMPARE_TENSOR(c_output.get(), output.get(), true);
// }
TEST_F(CPUTest, CPUMatmul1) {
    SETUP_OP(CPUMatmul, false, true, 4);
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
    PRINT_TENSOR_SHAPES(input0, input1, c_output, output);
    //    c_output->printData<float>();
    COMPARE_TENSOR(c_output.get(), output.get(), true);
}
// TEST_F(CPUTest, CPUMatmul3) {
//     SETUP_OP(CPUMatmul, true, false, 4);
//     TENSOR(input0);
//     TENSOR(input1);
//     TENSOR(output);
//     TENSOR(c_output);
//     TEST_LOAD(input0);
//     TEST_LOAD(input1);
//     TEST_LOAD(output);
//     TEST_RESHAPE({input0, input1}, {c_output});
//     PRINT_TENSOR_SHAPES(input0, input1, c_output, output);
//     TEST_SETUP({input0, input1}, {c_output});
//     TEST_EXCUTE({input0, input1}, {c_output});
////     c_output->printData<float>();
//     COMPARE_TENSOR(c_output.get(), output.get(), true);
// }