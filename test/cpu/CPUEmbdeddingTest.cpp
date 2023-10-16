//
// Created by lx on 23-10-15.
//
#include "CPUTest.hpp"
#include "backends/cpu/CPUEmbedding.hpp"

TEST_F(CPUTest, CPUEmbedding1) {
    SETUP_OP(CPUEmbedding, 128, 180);
    TENSOR(input0);
    TENSOR(output);
    TENSOR(p_output);
    p_output->setName("output");
    loader.load(input0);
    loader.load(p_output);
    op->reshape({input0}, {output});
    op->setUp({input0}, {output});
    loader.load(&op->weight_, false);
    op->execute({input0}, {output});
    COMPARE_TENSOR(p_output.get(), output.get());
}