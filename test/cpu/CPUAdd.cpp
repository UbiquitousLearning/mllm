//
// Created by 咸的鱼 on 2023/10/14.
//

#include "CPUTest.hpp"
#include "backends/cpu/CPUAdd.hpp"
#include "ParamLoader.hpp"

#define SETUP_OP(type_)                     \
    auto op = new type_(bn_, #type_, false); \
    auto loader = ParamLoader(#type_ +std::string(".mllm"))
#define TENSOR(name_) auto name_ = std::make_shared<Tensor>(bn_); name_->setName(#name_);
using namespace mllm;
TEST_F(CPUTest, CPUAdd1) {
    SETUP_OP(CPUAdd);
    TENSOR(input1);
    TENSOR(input2);
    TENSOR(output);
    input1->reshape({2, 2});
    input2->reshape({2, 2});
    op->reshape({input1, input2}, {output});
    EXPECT_EQ(output->shape(0), 2);
    EXPECT_EQ(output->shape(1), 2);
    EXPECT_EQ(output->shape(2), 1);
    EXPECT_EQ(output->shape(3), 1);
    op->setUp({input1, input2}, {output});
    loader.load(input1);
    loader.load(input2);
    op->execute({input1, input2}, {output});
    //TODO: check output?
    output->printData<float>();
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}