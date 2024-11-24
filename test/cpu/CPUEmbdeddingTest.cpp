//
// Created by Xiang Li on 23-10-15.
//
#include "CPUTest.hpp"
#include "backends/cpu/op/CPUEmbedding.hpp"

TEST_F(CPUTest, CPUEmbedding1) {
    SETUP_OP(CPUEmbedding, 128, 180, 4);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(p_output);
    p_output->setName("output");
    loader.load(input0);
    loader.load(p_output);
    op->reshape({input0}, {output});
    op->setUp({input0}, {output});
    op->load(loader);
    op->execute({input0}, {output});
    COMPARE_TENSOR(p_output.get(), output.get(), true);
}