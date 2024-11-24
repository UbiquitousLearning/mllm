//
// Created by Xiang Li on 23-10-16.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUScale.hpp"
TEST_F(CPUTest, CPUScale1) {
    SETUP_OP(CPUScale, 1);
    TENSOR(input0);
    TENSOR(output);
    input0->reshape(1, 2, 3, 4);

    TEST_RESHAPE({input0}, {output});
    TEST_SETUP({input0}, {output});
    input0->fullDataTest();
    TEST_EXCUTE({input0}, {output});
    COMPARE_TENSOR(input0.get(), output.get(), true);
}
TEST_F(CPUTest, CPUScale2) {
    SETUP_OP(CPUScale, 0, 1);
    TENSOR(input0);
    TENSOR(output);
    input0->reshape(1, 2, 3, 4);

    TEST_RESHAPE({input0}, {output});
    TEST_SETUP({input0}, {output});
    TEST_EXCUTE({input0}, {output});
    ASSERT_EQ(input0->batch(), output->batch());
    ASSERT_EQ(input0->head(), output->head());
    ASSERT_EQ(input0->sequence(), output->sequence());
    ASSERT_EQ(input0->dimension(), output->dimension());
    // for (int i = 0; i < output->count(); ++i) {
    //     ASSERT_EQ(*(output->hostPtr<float>() + i), 1.0) << "Data @" << i << " is not 1.0";
    // }
}
TEST_F(CPUTest, CPUScale3) {
    SETUP_OP(CPUScale, 2, 1);
    TENSOR(input0);
    TENSOR(output);
    input0->reshape(1, 2, 3, 4);
    TEST_RESHAPE({input0}, {output});
    TEST_SETUP({input0}, {output});
    TEST_EXCUTE({input0}, {output});
    ASSERT_EQ(input0->batch(), output->batch());
    ASSERT_EQ(input0->head(), output->head());
    ASSERT_EQ(input0->sequence(), output->sequence());
    ASSERT_EQ(input0->dimension(), output->dimension());
    // for (int i = 0; i < output->count(); ++i) {
    //     ASSERT_EQ(*(output->hostPtr<float>() + i), *(input0->hostPtr<float>() + i) * 2 + 1) << "Data @" << i << " is not 1.0";
    // }
}
TEST_F(CPUTest, CPUScale4) {
    SETUP_OP(CPUScale, 2, 1, false);
    TENSOR(input0);
    TENSOR(output);
    input0->reshape(1, 2, 3, 4);
    TEST_RESHAPE({input0}, {output});
    TEST_SETUP({input0}, {output});
    TEST_EXCUTE({input0}, {output});
    ASSERT_EQ(input0->batch(), output->batch());
    ASSERT_EQ(input0->head(), output->head());
    ASSERT_EQ(input0->sequence(), output->sequence());
    ASSERT_EQ(input0->dimension(), output->dimension());
    // for (int i = 0; i < output->count(); ++i) {
    //     ASSERT_EQ(*(output->hostPtr<float>() + i), (*(input0->hostPtr<float>() + i) + 1) * 2) << "Data @" << i << " is not 1.0";
    // }
}