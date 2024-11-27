//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUSPLITFUNC_HPP
#define CPUSPLITFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUsplitFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int size = args.size();
        std::vector<int> each_dims;
        for (int i = 0; i < size - 2; i++) {
            each_dims.push_back(args[i]);
        }
        Chl split_dim = (Chl)args[size - 2];
        int head_size = (int)args[size - 1];
        int split_num_ = each_dims.size();
        // store each dims
        int split_dim_size_ = 0;
        std::vector<int> each_dims_;
        for (size_t i = 0; i < each_dims.size(); ++i) {
            each_dims_.push_back((float)each_dims[i]);
            split_dim_size_ += each_dims[i];
        }
        assert(split_num_ == outputs.size());
        switch (split_dim) {
        case Chl::HEAD: {
            assert(inputs[0]->head() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            assert(inputs[0]->sequence() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            assert(inputs[0]->dimension() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::D_HD: {
            assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::HD: {
            assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            break;
        }
        }
        vector<shared_ptr<Tensor>> shared_outputs = {};
        for (const auto &output : outputs) {
            output->alloc();
            shared_outputs.push_back(std::shared_ptr<Tensor>(output, [](Tensor *) {}));
        }
        if (inputs[0]->masterTensor() == nullptr && !inputs[0]->childTensors().empty()) {
            inputs[0]->free();
        }
        inputs[0]->addTensors(shared_outputs, split_dim);
        //     for (const auto &output : outputs) {
        //         output->setDtype(MLLM_TYPE_F32);
        //         output->alloc();
        //     }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};

} // namespace mllm
#endif // CPUSPLITFUNC_HPP