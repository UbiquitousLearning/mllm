//
// Created by Xiang Li on 2023/10/14.
//

#include "CPUTest.hpp"
#include "backends/cpu/op/CPUAdd.hpp"
#include "TestLoader.hpp"

using namespace mllm;
TEST_F(CPUTest, CPUAdd1) {
    SETUP_OP(CPUAdd, 4);
    TENSOR(input0);
    TENSOR(input1);
    TENSOR(output);
    loader.load(input0);
    loader.load(input1);
    op->reshape({input0, input1}, {output});
    EXPECT_GE(output->batch(), 1);
    EXPECT_GE(output->head(), 1);
    EXPECT_GE(output->sequence(), 1);
    EXPECT_GE(output->dimension(), 1);
    op->setUp({input0, input1}, {output});
    op->execute({input0, input1}, {output});
    // TODO: check output?
    Tensor *torch_output = new Tensor(bn_);
    torch_output->setName("output");
    loader.load(torch_output, true);
    //    output->printData<float>();
    EXPECT_TRUE(isSame(output.get(), torch_output));
}
