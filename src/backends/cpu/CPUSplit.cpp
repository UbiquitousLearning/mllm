#include "CPUSplit.hpp"

namespace mllm {

CPUSplit::CPUSplit(Backend *bn,  string opName,  int splitNum, Chl splitDim, int splitDimSize, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    split_num_ = splitNum;
    split_dim_ = splitDim;
    split_dim_size_ =  splitDimSize;
}

ErrorCode CPUSplit::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(split_num_ == outputs.size());
    assert(inputs.size() == 1);
    switch (split_dim_) {
        case Chl::HEAD: {
            assert(inputs[0]->head() % split_num_== 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head() / split_num_, inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            assert(inputs[0]->sequence() % split_num_ == 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / split_num_, inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            assert(inputs[0]->dimension() % split_num_ == 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension() / split_num_);
            }
            break;
        }
        case D_HD: {
            assert(inputs[0]->dimension() % split_num_ == 0);
            for (auto &output: outputs) {
                output->reshape(inputs[0]->batch(), split_dim_size_, inputs[0]->sequence(), inputs[0]->dimension() / (split_num_*split_dim_size_));
            }
            break;
        }
        case HD: {
            assert(inputs[0]->dimension() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), split_dim_size_, inputs[0]->sequence(), inputs[0]->dimension() / (split_num_ * split_dim_size_));
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

    return Op::execute(inputs, outputs);
}

ErrorCode CPUSplit::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    return Op::setUp(inputs, outputs);
}
} // namespace mllm

