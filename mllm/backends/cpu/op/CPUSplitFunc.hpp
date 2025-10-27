//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUSPLITFUNC_HPP
#define CPUSPLITFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include "../compute/Split.hpp"
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUsplitFunction : public Op {
private:
    int thread_count = 4;
    std::vector<int> each_dims_;
    Chl split_dim_;
    int head_size_;

public:
    CPUsplitFunction(Backend *bn, string name, int threadCount,
                     const std::vector<int> &each_dims, Chl split_dim, int head_size) :
        Op(bn, name), thread_count(threadCount), each_dims_(each_dims),
        split_dim_(split_dim), head_size_(head_size) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int split_num_ = each_dims_.size();
        // store each dims
        int split_dim_size_ = 0;
        for (size_t i = 0; i < each_dims_.size(); ++i) {
            split_dim_size_ += each_dims_[i];
        }
        assert(split_num_ == outputs.size());
        switch (split_dim_) {
        case Chl::HEAD: {
            // assert(inputs[0]->head() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            // assert(inputs[0]->sequence() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            // assert(inputs[0]->dimension() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::D_HD: {
            // assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size_, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::HD: {
            // assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size_, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            break;
        }
        }
        if (inputs[0]->allowAggregated()) {
            vector<shared_ptr<Tensor>> shared_outputs = {};
            for (const auto &output : outputs) {
                output->alloc();
                shared_outputs.push_back(output);
            }
            if (inputs[0]->masterTensor() == nullptr && !inputs[0]->childTensors().empty()) {
                inputs[0]->free();
            }
            inputs[0]->addTensors(shared_outputs, split_dim_);
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int split_num_ = each_dims_.size();
        // store each dims
        int split_dim_size_ = 0;
        for (size_t i = 0; i < each_dims_.size(); ++i) {
            split_dim_size_ += each_dims_[i];
        }
        assert(split_num_ == outputs.size());
        switch (split_dim_) {
        case Chl::HEAD: {
            // assert(inputs[0]->head() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            // assert(inputs[0]->sequence() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            // assert(inputs[0]->dimension() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::D_HD: {
            // assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size_, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::HD: {
            // assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size_, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            break;
        }
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // This path is taken only if the memory aggregation in setUp was not performed.
        if (inputs[0]->aggregatedTensors().empty()) {
            std::vector<void *> out_pointers;
            std::vector<DataType> out_types;

            assert(each_dims_.size() == outputs.size());
            for (const auto &output : outputs) {
                if (output->hostPtr<void>() == nullptr) {
                    output->alloc();
                }
                if (output->dtype() == MLLM_TYPE_F32) {
                    out_pointers.push_back(output->ptrAt<float>(0, 0, 0, 0));
                } else if (output->dtype() == MLLM_TYPE_F16) {
                    out_pointers.push_back(output->ptrAt<mllm_fp16_t>(0, 0, 0, 0));
                }
                out_types.push_back(output->dtype());
            }
            const int origin_dims[4] = {inputs[0]->batch(), inputs[0]->sequence(), inputs[0]->head(), inputs[0]->dimension()};

            efficient_split(inputs[0]->ptrAt<float>(0, 0, 0, 0),
                            origin_dims,
                            out_pointers,
                            out_types,
                            each_dims_,
                            split_dim_);
        }

        return MLLM_NO_ERROR;
    }
};

class CPUsplitFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam is structured to pass the split parameters.
        // Example: {"num_splits": 2, "dim_0": 64, "dim_1": 64, "split_dim": 3, "head_size": 12}
        int num_splits = static_cast<int>(op_param.at("num_splits"));
        std::vector<int> each_dims;
        for (int i = 0; i < num_splits; ++i) {
            each_dims.push_back(static_cast<int>(op_param.at("dim_" + std::to_string(i))));
        }
        Chl split_dim = (Chl)op_param.at("split_dim");
        int head_size = static_cast<int>(op_param.at("head_size"));

        return new CPUsplitFunction(bn, name, threadCount, each_dims, split_dim, head_size);
    }
};

} // namespace mllm
#endif // CPUSPLITFUNC_HPP