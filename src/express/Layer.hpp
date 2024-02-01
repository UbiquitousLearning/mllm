//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <utility>

#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"

#include <Module.hpp>

namespace mllm {

class Layer {
public:
    Layer() = default;
    void init(std::string name, OpType type) {
        name_ = std::move(name);
        param_["type"] = type;
        backend_ = Module::backends[MLLM_CPU];
        // constexpr int threadCount = 4;
        // op_ = backend_->opCreate(param_, std::move(name), threadCount);
        // op_->load(*Module::loader);
    }
    //    Tensor &operator()(Tensor &input) {
protected:
    Tensor &_1I1O_OP(Tensor &input) {
        if (op_ == nullptr) {
            constexpr int threadCount = 4;
            op_ = backend_->opCreate(param_, name_, threadCount);
            op_->load(*Module::loader);
        }

        string next_name = "out-" + op_->name();
        switch (input.status()) {
        case TENSOR_STATIC_INIT: {
            if (Tensor::gph_.find(input.name()) == Tensor::gph_.end()) {
                Tensor::gph_[input.name()] = input;
                Tensor::gph_[input.name()].setName(input.name());
            }
            if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                Tensor::gph_[next_name] = Tensor(backend_);
                Tensor::gph_[next_name].setName(next_name);
            }
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->reshape(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        case TENSOR_STATIC_SHAPED: {
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->setUp(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        case TENSOR_STATIC_ALLOCED: {
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->execute(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        default: {
            break;
        }
        }
        Tensor::gph_[next_name].status() = Tensor::gph_[input.name()].status();
        return Tensor::gph_[next_name];
    }
    Tensor &_2I1O_OP(Tensor &input0, Tensor &input1) {
        if (op_ == nullptr) {
            constexpr int threadCount = 4;
            op_ = backend_->opCreate(param_, name_, threadCount);
            op_->load(*Module::loader);
        }

        string next_name = "out-" + op_->name();
        assert(input0.status() == input1.status());
        switch (input0.status()) {
        case TENSOR_STATIC_INIT: {
            if (Tensor::gph_.find(input0.name()) == Tensor::gph_.end()) {
                Tensor::gph_[input0.name()] = input0;
                Tensor::gph_[input0.name()].setName(input0.name());
            }
            if (Tensor::gph_.find(input1.name()) == Tensor::gph_.end()) {
                Tensor::gph_[input1.name()] = input1;
                Tensor::gph_[input1.name()].setName(input1.name());
            }
            if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                Tensor::gph_[next_name] = Tensor(backend_);
                Tensor::gph_[next_name].setName(next_name);
            }
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input0.name()]),
                                                     std::make_shared<Tensor>(Tensor::gph_[input1.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->reshape(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        case TENSOR_STATIC_SHAPED: {
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input0.name()]),
                                                     std::make_shared<Tensor>(Tensor::gph_[input1.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->setUp(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        case TENSOR_STATIC_ALLOCED: {
            vector<shared_ptr<Tensor>> shared_inputs{std::make_shared<Tensor>(Tensor::gph_[input0.name()]),
                                                     std::make_shared<Tensor>(Tensor::gph_[input1.name()])};
            vector<shared_ptr<Tensor>> shared_outputs{std::make_shared<Tensor>(Tensor::gph_[next_name])};
            op_->execute(shared_inputs, shared_outputs);
            Tensor::gph_[next_name] = *shared_outputs[0];
            break;
        }
        default: {
            break;
        }
        }
        Tensor::gph_[next_name].status() = Tensor::gph_[input0.name()].status();
        return Tensor::gph_[next_name];
    }

    std::string name_;
    Op *op_ = nullptr;
    Backend *backend_{};
    OpParam param_;
};

class Linear final : public Layer {
public:
    explicit Linear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LINEAR);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class SiLU final : public Layer {
public:
    SiLU() = default;
    SiLU(std::string name) {
        init(std::move(name), OpType::SILU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Softmax final : public Layer {
public:
    explicit Softmax(Chl axis, std::string name) {
        param_["axis"] = axis;
        init(std::move(name), OpType::SOFTMAX);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Embedding final : public Layer {
public:
    explicit Embedding(int vocab_size, int hidden_size, std::string name) {
        param_["hidden_size"] = hidden_size;
        param_["vocab_size"] = vocab_size;
        init(std::move(name), OpType::EMBEDDING);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Causalmask final : public Layer {
public:
    explicit Causalmask(std::string name) {
        init(std::move(name), OpType::CAUSALMASK);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RoPE final : public Layer {
public:
    explicit RoPE(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class KVCache final : public Layer {
public:
    explicit KVCache(int cache_max, std::string name) {
        param_["n_rep"] = 1;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RMSNorm final : public Layer {
public:
    explicit RMSNorm(int norm_size, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        init(std::move(name), OpType::RMSNORM);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};


class Matmul final : public Layer {
public:
    explicit Matmul(bool transpose0, bool transpose1,  std::string name) {
        param_["transpose0"] = transpose0;
        param_["transpose1"] = transpose1;
        init(std::move(name), OpType::MATMUL);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }
};

} // namespace mllm

#endif // OPERATION_H
