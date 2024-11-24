#include "CPUSplit.hpp"

namespace mllm {

CPUSplit::CPUSplit(Backend *bn, string opName, int splitNum, Chl splitDim, int splitDimSize, int threadCount, std::vector<int> each_dims) :
    thread_count(threadCount), split_num_(splitNum), split_dim_(splitDim), split_dim_size_(splitDimSize), each_dims_(each_dims), Op(bn, opName) {
}

ErrorCode CPUSplit::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(split_num_ == outputs.size());
    assert(inputs.size() == 1);
    switch (split_dim_) {
    case Chl::HEAD: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int head_sum = 0;
                for (auto item : each_dims_) head_sum += item;
                assert(head_sum == inputs[0]->head() && "sum(each_dims_) miss match inputs[0]'s head dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        default: /*split for same size*/ {
            assert(inputs[0]->head() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head() / split_num_, inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        }
        break;
    }
    case Chl::SEQUENCE: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int seq_sum = 0;
                for (auto item : each_dims_) seq_sum += item;
                assert(seq_sum == inputs[0]->sequence() && "sum(each_dims_) miss match inputs[0]'s sequence dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        default: {
            assert(inputs[0]->sequence() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / split_num_, inputs[0]->dimension());
            }
            break;
        }
        }
        break;
    }
    case Chl::DIMENSION: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int dimension_sum = 0;
                for (auto item : each_dims_) dimension_sum += item;
                assert(dimension_sum == inputs[0]->sequence() && "sum(each_dims_) miss match inputs[0]'s dimension dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            assert(inputs[0]->dimension() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension() / split_num_);
            }
            break;
        }
        }
        break;
    }
    case Chl::D_HD: {
        assert(inputs[0]->dimension() % split_num_ == 0);
        for (auto &output : outputs) {
            output->reshape(inputs[0]->batch(), split_dim_size_, inputs[0]->sequence(), inputs[0]->dimension() / (split_num_ * split_dim_size_));
        }
        break;
    }
    case Chl::HD: {
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
