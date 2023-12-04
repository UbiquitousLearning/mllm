#include "CPUSplit.hpp"

namespace mllm {

CPUSplit::CPUSplit(Backend *bn,  string opName,  int splitNum, Chl splitDim, bool multiThread) :
    Op(bn, opName) {
    split_num_ = splitNum;
    split_dim_ = splitDim;
}

ErrorCode CPUSplit::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUSplit  reshape" << std::endl;
    CHECK_EQ(split_num_, outputs.size());
    CHECK_EQ(inputs.size(), 1);
    switch (split_dim_) {
        case Chl::HEAD: {
            CHECK_EQ(inputs[0]->head() % split_num_, 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head() / split_num_, inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            CHECK_EQ(inputs[0]->sequence() % split_num_, 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / split_num_, inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            CHECK_EQ(inputs[0]->dimension() % split_num_, 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension() / split_num_);
            }
            break;
        }
        default: {
            break;
        }
    }
    inputs[0]->addTensors(outputs, split_dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSplit::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUSplit()" << std::endl;
    return Op::execute(inputs, outputs);
}

ErrorCode CPUSplit::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUSplit() setUp" << std::endl;
    return Op::setUp(inputs, outputs);
}
} // namespace mllm

