//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include "Context.hpp"
#include "OpDefined.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Trace.hpp"
#include "Types.hpp"

#include <Module.hpp>

#include <regex>
#include <string>
#include <vector>

namespace mllm {

class Layer {
public:
    Layer() = default;
    ~Layer() {
        delete op_; // 手动添加 delete
        op_ = nullptr;
    }
    void init(std::string name, OpType type) {
        name_ = std::move(name);
        type_ = type;
        param_["type"] = type;
        Module::initBackend(MLLM_CPU);
        backend_ = Backend::global_backends[MLLM_CPU].get();
        saved_list_idx = Module::listIdx;
        init_ = true;
    }
    bool ready() {
        return init_;
    }

    /**** for forward ****/
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input0, Tensor input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input0, Tensor input1, Tensor input2) {
        auto ts = run({input0, input1, input2}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input0, Tensor input1, Tensor input2, Tensor input3) {
        auto ts = run({input0, input1, input2, input3}, 1);
        return ts[0];
    }

    /**** for dynamic load ****/
    void load(std::shared_ptr<AbstructLoader> loader = nullptr) {
        if (op_ == nullptr) {
            initOp();
        }
        if (inited_loaded && loaded_param)
            return;
        if (!loader) {
            loader = Module::llm_model_ptr->loader;
        }
        op_->load(*loader);
        loaded_param = true;
        inited_loaded = true;
    }
    bool &loaded() {
        return loaded_param;
    }
    void free() {
        op_->free({}, {});
        loaded_param = false;
        inited_loaded = false;
    }

    /**** for change backend ****/
    void to(BackendType type) {
        if (type == backend_->type()) {
            return;
        }
        Module::initBackend(type);
        backend_ = Backend::global_backends[type].get();
        if (!init_) {
            init(name_, type_);
        }
    }
    void cpu() {
        to(MLLM_CPU);
    }
    void cl() {
#ifdef USE_OPENCL
        to(MLLM_OPENCL);
#else
        throw std::runtime_error("OpenCL backend is not available. Please compile with USE_OPENCL=ON.");
#endif
    }

protected:
    // 统一入口
    vector<Tensor> run(vector<Tensor> inputs, int N = 1) {
        //////////==============QNN only====================///////////
        if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) {
            auto backend = Backend::global_backends[MLLM_QNN].get();
            return backend->runLayer(this, inputs, N);
        }
        //////////==============QNN only====================///////////

        // auto start_time = mllm_time_us();
        // ==================== [开始] op ====================//
        Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
        /*** part.1 change backend(if) ***/
        if (module && module->doChangeBn) {
            if (check_op_support(Backend::global_backends[module->device()].get())) {
                to(module->device());
            } else {
                std::cout << "Backend " << module->device() << " does not support the operation, falling back to CPU." << std::endl;
                to(MLLM_CPU);
            }
        }
        /*** part.2 init op(if) ***/
        if (op_ == nullptr || op_->backend()->type() != backend_->type()) {
            auto created = initOp();
            assert(op_ != nullptr && "Op creation failed, please check the backend and parameters.");
        }
        /*** part.3 load params(if) ***/
        if (module && (module->doLoad || !inited_loaded) && !module->doChangeBn) { // load
            if (module->doLoad) {
                op_->load(*module->loader);
                inited_loaded = true;
            } else if (loaded_param) {
                inited_loaded = loaded_param;
            } else if (!inited_loaded) {
                ParamLoader empty_loader("");
                op_->load(empty_loader);
                inited_loaded = true;
            }
        }
        /*** part.4 forward ***/
        auto backend = backend_;
        if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) {
            backend = Backend::global_backends[MLLM_QNN].get();
        }
        for (auto &input : inputs) {
            if (input.backend() != backend) {
                input.to(backend->type());
            }
        }
        vector<string> out_names;
        int count = (N > 1) ? N : 1;
        for (int i = 0; i < count; ++i) {
            std::string tensor_name = (N > 1) ? "out-" + op_->name() + "-" + std::to_string(i) : "out-" + op_->name();
            out_names.push_back(tensor_name);
        }
        // ==================== [结束] op ====================//
        // if (module && !module->doTrace) {
        //     auto end_time = mllm_time_us();
        //     std::cout << name_ << " dispatch Layer in " << (end_time - start_time) / 1000.0F << " ms" << std::endl;
        // }
        return backend->runOp(op_, inputs, out_names, false);
    }

public:
    std::string name_;
    OpType type_;
    Op *op_ = nullptr;
    Backend *backend_{};
    OpParam param_;
    bool init_ = false;
    int saved_list_idx;

    bool inited_loaded = false;
    bool loaded_param = false;

    static map<string, string> layername_2_tensorname;
    static bool use_layername_2_tensorname;

private:
    bool initOp(Backend *bn = nullptr) { // 返回bn是否能成功创建op
        assert(op_ == nullptr);
        if (bn == nullptr) {
            bn = backend_;
        }
        if ((param_["type"] == KVCACHE || param_["type"] == KVCACHENPU)
            && (Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end())) { // 针对qnn专门处理
            if (kv_cache_map.find(name_) == kv_cache_map.end()) {
                // for the prefill part, we need to create a new op
                param_["type"] = KVCACHENPU;
                assert(bn->type() == MLLM_CPU && "QNN backend is required for cpu's KVCACHENPU.");
                op_ = bn->opCreate(param_, name_);
                kv_cache_map[name_] = op_;
            } else {
                // for the decoding part, we need to get created op from global container
                op_ = kv_cache_map[name_];
            }
            op_->type() = (OpType)param_["type"];
            return true;
        }
        if (!(op_ = bn->opCreate(param_, name_))) {
            // std::cout << "Backend" << op_->type() << " does not support the operation, falling back to CPU." << std::endl; // ttt
            op_ = backend_->opCreate(param_, name_); // cpu fallback
            op_->type() = (OpType)param_["type"];
            return false;
        }
        op_->type() = (OpType)param_["type"];
        return true;
    }
    bool check_op_support(Backend *bn) {
        return bn->opCreate(param_, name_);
    }
};

class Linear final : public Layer {
public:
    explicit Linear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LINEAR);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class HeadLinear final : public Layer {
public:
    explicit HeadLinear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::HEADLINEAR);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class SparseIdLinear final : public Layer {
public:
    SparseIdLinear(int in_dim, int out_dim, std::string name) {
        param_["in_dim_"] = (float)in_dim;
        param_["out_dim_"] = (float)out_dim;
        init(std::move(name), OpType::SPARSEIDLINEAR);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class SparseLinear final : public Layer {
public:
    SparseLinear(int in_dim, int out_dim, std::string name) {
        param_["in_dim_"] = (float)in_dim;
        param_["out_dim_"] = (float)out_dim;
        init(std::move(name), OpType::SPARSELINEAR);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Predictor final : public Layer {
public:
    Predictor(int in_dim, int out_dim, std::string name) {
        param_["in_dim"] = (float)in_dim;
        param_["out_dim"] = (float)out_dim;
        init(std::move(name), OpType::PREDICTOR);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class ElasticLinear final : public Layer {
public:
    ElasticLinear() = default;
    explicit ElasticLinear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::ELASTICLINEAR);
    }
    Tensor operator()(Tensor input0, int activate_input_dim, int activate_output_dim) {
        auto activate_input_dim_tensor = Tensor(activate_input_dim, backend_);
        auto activate_output_dim_tensor = Tensor(activate_output_dim, backend_);
        auto ts = run({input0, activate_input_dim_tensor, activate_output_dim_tensor}, 1);
        return ts[0];
    }
};

class ShadowLinear final : public Layer {
public:
    ShadowLinear() = default;
    explicit ShadowLinear(int in_features, int out_features, int max_position, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["max_position"] = max_position;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LINEARINT8SHADOW);
    }
    Tensor operator()(Tensor input0, Tensor input1, Tensor input2) {
        auto ts = run({input0, input1, input2}, 1);
        return ts[0];
    }
};

class SiLU final : public Layer {
public:
    SiLU() = default;
    SiLU(std::string name) {
        init(std::move(name), OpType::SILU);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class ReLU final : public Layer {
public:
    ReLU() = default;
    ReLU(std::string name) {
        init(std::move(name), OpType::RELU);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class ReLUSquaredActivation final : public Layer {
public:
    ReLUSquaredActivation() = default;
    ReLUSquaredActivation(std::string name) {
        init(std::move(name), OpType::RELU2);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class GELU final : public Layer {
public:
    GELU() = default;
    GELU(std::string name) {
        init(std::move(name), OpType::OP_GELU);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class QuickGELU final : public Layer {
public:
    QuickGELU() = default;
    explicit QuickGELU(std::string name) {
        init(std::move(name), OpType::QUICKGLUE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

using ActFnConstructor = std::function<Layer(const std::string &)>;
inline std::map<std::string, ActFnConstructor> ACT_FN = {
    {"SiLU", [](const std::string &name) { return SiLU(name); }},
    {"ReLU", [](const std::string &name) { return ReLU(name); }},
    {"ReLU2", [](const std::string &name) { return ReLUSquaredActivation(name); }},
    {"GELU", [](const std::string &name) { return GELU(name); }},
    {"QuickGELU", [](const std::string &name) { return QuickGELU(name); }},
};

class Softmax final : public Layer {
public:
    Softmax() = default;
    explicit Softmax(Chl axis, std::string name) {
        param_["axis"] = axis;
        init(std::move(name), OpType::SOFTMAX);
    }
    explicit Softmax(Chl axis, bool do_causal_mask, std::string name) {
        param_["axis"] = axis;
        param_["do_causal_mask"] = do_causal_mask;
        init(std::move(name), OpType::SOFTMAX);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input, int axis_classes) {
        auto axis_classes_tensor = Tensor(axis_classes, backend_);
        auto ts = run({input, axis_classes_tensor}, 1);
        return ts[0];
    }
};
class Sigmoid final : public Layer {
public:
    Sigmoid() = default;
    explicit Sigmoid(std::string name) {
        init(std::move(name), OpType::SIGMOID);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Embedding final : public Layer {
public:
    explicit Embedding(int vocab_size, int hidden_size, std::string name) {
        param_["hidden_size"] = hidden_size;
        param_["vocab_size"] = vocab_size;
        init(std::move(name), OpType::EMBEDDING);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Causalmask final : public Layer {
public:
    Causalmask() = default;
    explicit Causalmask(std::string name) {
        init(std::move(name), OpType::CAUSALMASK);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input0, int kvcache_seq) {
        auto kvcache_seq_tensor = Tensor(kvcache_seq, backend_);
        auto ts = run({input0, kvcache_seq_tensor}, 1);
        return ts[0];
    }
};

class CausalTreeMask final : public Layer {
public:
    CausalTreeMask() = default;
    explicit CausalTreeMask(std::string name) {
        init(std::move(name), OpType::CAUSALTREEMASK);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    Tensor operator()(Tensor input0, int kvcache_seq, Tensor tree_ancestor) {
        auto kvcache_seq_tensor = Tensor(kvcache_seq, backend_);
        auto ts = run({input0, kvcache_seq_tensor, tree_ancestor}, 1);
        return ts[0];
    }
};

class SlidingWindowMask final : public Layer {
public:
    explicit SlidingWindowMask(int window_size, std::string name) {
        param_["window_size"] = window_size;
        init(std::move(name), OpType::SLIDINGWINDOWMASK);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

typedef std::unordered_map<string, std::any> RoPEConfig;

class RoPE final : public Layer {
public:
    RoPE() = default;

    explicit RoPE(int pose_type, const RoPEConfig &config, std::string name) {
        param_["pose_type"] = pose_type;
        auto it_rope_theta = config.find("rope_theta");
        if (it_rope_theta != config.end()) {
            param_["rope_theta"] = std::any_cast<float>(it_rope_theta->second);
        }

        auto it_max_position_embeddings = config.find("max_position_embeddings");
        if (it_max_position_embeddings != config.end()) {
            param_["max_position_embeddings"] = std::any_cast<int>(it_max_position_embeddings->second);
        }

        auto it_partial_rotary_factor = config.find("partial_rotary_factor");
        if (it_partial_rotary_factor != config.end()) {
            param_["partial_rotary_factor"] = std::any_cast<float>(it_partial_rotary_factor->second);
        }

        if (config.find("rope_scaling") != config.end()) {
            auto rope_scaling = std::any_cast<map<string, std::any>>(config.at("rope_scaling"));
            auto it = rope_scaling.find("rope_type");
            if (it != rope_scaling.end()) {
                string rope_type = std::any_cast<string>(it->second);
                if (rope_type == "default") {
                    param_["rope_type"] = DEFAULT;
                } else if (rope_type == "llama3") {
                    param_["rope_type"] = LLAMA3;
                    param_["factor"] = std::any_cast<float>(rope_scaling.at("factor"));
                    param_["high_freq_factor"] = std::any_cast<float>(rope_scaling.at("high_freq_factor"));
                    param_["low_freq_factor"] = std::any_cast<float>(rope_scaling.at("low_freq_factor"));
                    param_["original_max_position_embeddings"] = std::any_cast<int>(rope_scaling.at("original_max_position_embeddings"));
                } else {
                    std::cout << "[TODO]rope type " << rope_type << " not support!!!!" << std::endl;
                }
            }
        }

        init(std::move(name), OpType::ROPE);
    }

    explicit RoPE(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPE);
    }
    explicit RoPE(int pose_type, float rope_theta, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        init(std::move(name), OpType::ROPE);
    }
    explicit RoPE(int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        param_["partial_rotary_factor"] = partial_rotary_factor;
        init(std::move(name), OpType::ROPE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    void clearCache() {
        return op_->clearCache();
    }
};

class RoPETree final : public Layer {
public:
    RoPETree() = default;

    explicit RoPETree(int pose_type, const RoPEConfig &config, std::string name) {
        param_["pose_type"] = pose_type;
        auto it_rope_theta = config.find("rope_theta");
        if (it_rope_theta != config.end()) {
            param_["rope_theta"] = std::any_cast<float>(it_rope_theta->second);
        }

        auto it_max_position_embeddings = config.find("max_position_embeddings");
        if (it_max_position_embeddings != config.end()) {
            param_["max_position_embeddings"] = std::any_cast<int>(it_max_position_embeddings->second);
        }

        auto it_partial_rotary_factor = config.find("partial_rotary_factor");
        if (it_partial_rotary_factor != config.end()) {
            param_["partial_rotary_factor"] = std::any_cast<float>(it_partial_rotary_factor->second);
        }

        if (config.find("rope_scaling") != config.end()) {
            auto rope_scaling = std::any_cast<map<string, std::any>>(config.at("rope_scaling"));
            auto it = rope_scaling.find("rope_type");
            if (it != rope_scaling.end()) {
                string rope_type = std::any_cast<string>(it->second);
                if (rope_type == "default") {
                    param_["rope_type"] = DEFAULT;
                } else if (rope_type == "llama3") {
                    param_["rope_type"] = LLAMA3;
                    param_["factor"] = std::any_cast<float>(rope_scaling.at("factor"));
                    param_["high_freq_factor"] = std::any_cast<float>(rope_scaling.at("high_freq_factor"));
                    param_["low_freq_factor"] = std::any_cast<float>(rope_scaling.at("low_freq_factor"));
                    param_["original_max_position_embeddings"] = std::any_cast<int>(rope_scaling.at("original_max_position_embeddings"));
                } else {
                    std::cout << "[TODO]rope type " << rope_type << " not support!!!!" << std::endl;
                }
            }
        }

        init(std::move(name), OpType::ROPETREE);
    }

    explicit RoPETree(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPETREE);
    }
    explicit RoPETree(int pose_type, float rope_theta, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        init(std::move(name), OpType::ROPETREE);
    }
    explicit RoPETree(int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        param_["partial_rotary_factor"] = partial_rotary_factor;
        init(std::move(name), OpType::ROPETREE);
    }
    Tensor operator()(Tensor input, Tensor tree_ancestor) {
        auto ts = run({input, tree_ancestor}, 1);
        return ts[0];
    }
    void clearCache() {
        return op_->clearCache();
    }
};

class RoPESimple final : public Layer {
public:
    RoPESimple() = default;
    explicit RoPESimple(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPESIMPLE);
    }
};

class IRoPE final : public Layer {
public:
    IRoPE() = default;
    explicit IRoPE(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::IROPE);
    }
    explicit IRoPE(int pose_type, float rope_theta, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        init(std::move(name), OpType::IROPE);
    }
    explicit IRoPE(int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        param_["partial_rotary_factor"] = partial_rotary_factor;
        init(std::move(name), OpType::IROPE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    void clearCache() {
        return op_->clearCache();
    }
};

class KVCache final : public Layer {
public:
    KVCache() = default;
    explicit KVCache(int head, int hidden, int n_rep, int cache_max, std::string name) {
        param_["head"] = head;
        param_["hidden"] = hidden;
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        init(std::move(name), OpType::KVCACHE);
    }

    explicit KVCache(int head, int hidden, int n_rep, int cache_max, string attn_impl, std::string name) {
        param_["head"] = head;
        param_["hidden"] = hidden;
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        param_["fa2"] = (attn_impl == "flash_attention_2" || attn_impl == "sage_attention");
        if (attn_impl == "sage_attention" && hidden % QK8_0F == 0 && KVCacheSageDtypeBit == 8) {
            init(std::move(name), OpType::KVCACHESAGE);
        } else {
            init(std::move(name), OpType::KVCACHE);
        }
    }

    explicit KVCache(int cache_max, std::string name) {
        param_["n_rep"] = 1;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, bool for_xnn, std::string name) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = for_xnn;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, std::string name, bool npuEnbaled) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        if (npuEnbaled) {
            init(std::move(name), OpType::KVCACHENPU);
        } else {
            init(std::move(name), OpType::KVCACHE);
        }
    }
    explicit KVCache(int head, int hidden, int n_rep, int cache_max, std::string name, bool npuEnbaled) {
        param_["head"] = head;
        param_["hidden"] = hidden;
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        param_["for_xnn"] = false;
        if (npuEnbaled) {
            init(std::move(name), OpType::KVCACHENPU);
        } else {
            init(std::move(name), OpType::KVCACHE);
        }
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
    int getCacheSeqLen() {
        if (op_ == nullptr) {
            return -1;
        }
        return op_->getCacheSeqLen();
    }
    void clearCache() {
        if (op_ == nullptr) {
            return;
        }
        return op_->clearCache();
    }
};

class LayerNorm final : public Layer {
public:
    explicit LayerNorm(int norm_size, bool bias, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LAYERNORM);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class RMSNorm final : public Layer {
public:
    explicit RMSNorm(int norm_size, float epsilon, std::string name, bool isFP32 = true) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["isFP32"] = (float)isFP32;
        init(std::move(name), OpType::RMSNORM);
    }

    explicit RMSNorm(int norm_size, float epsilon, bool add_unit_offset, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["add_unit_offset"] = (float)add_unit_offset;
        init(std::move(name), OpType::RMSNORM);
    }

    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Matmul final : public Layer {
public:
    explicit Matmul(bool transpose0, bool transpose1, std::string name) {
        param_["transpose0"] = transpose0;
        param_["transpose1"] = transpose1;
        init(std::move(name), OpType::MATMUL);
    }
    Tensor operator()(Tensor input0, Tensor input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0];
    }
};

class Convolution2D final : public Layer {
public:
    explicit Convolution2D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_h"] = (float)kernal[0];
        param_["kernal_w"] = (float)kernal[1];
        param_["stride_h"] = (float)stride[0];
        param_["stride_w"] = (float)stride[1];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION2D);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Convolution3D final : public Layer {
public:
    explicit Convolution3D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_t"] = (float)kernal[0];
        param_["kernal_h"] = (float)kernal[1];
        param_["kernal_w"] = (float)kernal[2];
        param_["stride_t"] = (float)stride[0];
        param_["stride_h"] = (float)stride[1];
        param_["stride_w"] = (float)stride[2];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION3D);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class VisionRoPE final : public Layer {
public:
    explicit VisionRoPE(int dim_size, int spatial_merge_size, std::string name) {
        param_["dim"] = (float)dim_size;
        param_["spatial_merge_size"] = (float)spatial_merge_size;
        init(std::move(name), OpType::VISIONROPE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class VisionRoPESin final : public Layer {
public:
    explicit VisionRoPESin(int dim_size, int spatial_merge_size, std::string name) {
        param_["dim"] = (float)dim_size;
        param_["spatial_merge_size"] = (float)spatial_merge_size;
        init(std::move(name), OpType::VISIONROPESIN);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class VisionRoPECos final : public Layer {
public:
    explicit VisionRoPECos(int dim_size, int spatial_merge_size, std::string name) {
        param_["dim"] = (float)dim_size;
        param_["spatial_merge_size"] = (float)spatial_merge_size;
        init(std::move(name), OpType::VISIONROPECOS);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class MultimodalRoPE final : public Layer {
public:
    MultimodalRoPE() = default;
    explicit MultimodalRoPE(float rope_theta, int max_position_embeddings, vector<int> mrope_section, std::string name) {
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        for (int i = 0; i < mrope_section.size(); i++) {
            param_["mrope_section_" + std::to_string(i)] = (float)mrope_section[i];
        }
        if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) {
            init(std::move(name), OpType::MULTIMODALROPEPIP);
        } else {
            init(std::move(name), OpType::MULTIMODALROPE);
        }
    }
    Tensor operator()(Tensor input, Tensor &position_ids) {
        auto ts = run({input, position_ids}, 1);
        return ts[0];
    }
    void clearCache() {
        return op_->clearCache();
    }
};

class Parameter final : public Layer {
public:
    Parameter() = default;
    explicit Parameter(int batch, int seq, int head, int dim, std::string name) {
        param_["batch"] = batch;
        param_["seq"] = seq;
        param_["head"] = head;
        param_["dim"] = dim;
        init(std::move(name), OpType::PARAMETER);
    }
    Tensor operator()() {
        auto ts = run({}, 1);
        return ts[0];
    }
};

class Position final : public Layer {
public:
    explicit Position(std::string name) {
        init(std::move(name), OpType::POSITION);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

//  Only for QNN START

class SiLU_Full_Precision final : public Layer {
public:
    SiLU_Full_Precision() = default;
    SiLU_Full_Precision(std::string name) {
        init(std::move(name), OpType::SILU_FULL_PRECISION);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Quantize final : public Layer {
public:
    explicit Quantize(bool isNSHD, std::string name, DataType type = MLLM_TYPE_I8) {
        param_["isNSHD"] = (float)isNSHD;
        param_["dtype"] = (float)type;
        init(std::move(name), OpType::QUANTIZE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Direct final : public Layer {
public:
    enum DirectType : uint32_t {
        Normal = 0,
        ExternalInput = 1,
        ExternalOutput = 2,
        KeepLive = 3,
    };

    Direct(DirectType t, const std::string &name) {
        param_["DirectType"] = (float)t;
        init(name, OpType::DIRECT);
    }
};

class Dequantize final : public Layer {
public:
    explicit Dequantize(bool isNSHD, std::string name, bool isFP32 = true, DataType inType = MLLM_TYPE_I8) {
        param_["isNSHD"] = (float)isNSHD;
        param_["isFP32"] = (float)isFP32;
        param_["inType"] = (float)inType;
        init(std::move(name), OpType::DEQUANTIZE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class DequantizeAdd final : public Layer {
public:
    explicit DequantizeAdd(bool isNSHD, int out_features, std::string name, bool isFP32 = true, DataType inType = MLLM_TYPE_I8) {
        param_["isNSHD"] = (float)isNSHD;
        param_["out_features"] = out_features;
        param_["isFP32"] = (float)isFP32;
        param_["inType"] = (float)inType;
        init(std::move(name), OpType::DEQUANTIZEADD);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Dispatch final : public Layer {
public:
    explicit Dispatch(const std::string &name) {
        init(name, OpType::DISPATCH);
    }

    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Add final : public Layer {
public:
    explicit Add(std::string name) {
        init(std::move(name), OpType::ADD);
    }
    Tensor operator()(Tensor input0, Tensor input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0];
    }
};

class Mul final : public Layer {
public:
    explicit Mul(std::string name) {
        init(std::move(name), OpType::MUL);
    }
    Tensor operator()(Tensor input0, Tensor input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0];
    }
};

class View final : public Layer {
public:
    explicit View(int batch, int head, int seq, int dim, std::string name) {
        vector<int> dims;
        vector<int> data_dims;
        if (batch == -1 & seq == -1 & head != -1 & dim != -1) { // keep b&s change h&d
            if (head != 1) {
                dims = {batch, head, seq, -1};
                data_dims = {BATCH, DIMENSION, SEQUENCE, DIMENSION};
            } else {
                dims = {batch, -1, seq, -1};
                data_dims = {BATCH, -1, SEQUENCE, HEAD + DIMENSION};
            }
        } else if (batch == -1 & dim == -1 & head != -1 & seq != -1) { // keep b&d change h&s
            if (head != 1) {
                dims = {batch, head, -1, dim};
                data_dims = {BATCH, SEQUENCE, SEQUENCE, DIMENSION};
            } else {
                dims = {batch, -1, -1, dim};
                data_dims = {BATCH, -1, HEAD + SEQUENCE, DIMENSION};
            }
        } else if (head == -1 & dim == -1 & batch != -1 & seq != -1) { // keep h&d change b&s
            if (seq != 1) {
                dims = {-1, head, seq, dim};
                data_dims = {BATCH, HEAD, BATCH, DIMENSION};
            } else {
                dims = {-1, head, -1, dim};
                data_dims = {BATCH + SEQUENCE, HEAD, -1, DIMENSION};
            }
        } else if (batch != -1 & dim != -1 & head != -1 & seq != -1) { // change all dimension.

            dims = {batch, head, seq, dim};
            data_dims = {BATCH, HEAD, SEQUENCE, DIMENSION};

        } else {
            std::cout << "ERROR: " << name << " view [" << batch << ", " << head << ", " << seq << ", " << dim << "]" << std::endl;
        }
        param_["dim0"] = dims[0];
        param_["dim1"] = dims[1];
        param_["dim2"] = dims[2];
        param_["dim3"] = dims[3];
        param_["data_dim0"] = data_dims[0];
        param_["data_dim1"] = data_dims[1];
        param_["data_dim2"] = data_dims[2];
        param_["data_dim3"] = data_dims[3];
        init(std::move(name), OpType::VIEW);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class Transpose final : public Layer {
public:
    explicit Transpose(std::vector<int> perm, std::string name) {
        param_["perm0"] = perm[0];
        param_["perm1"] = perm[1];
        param_["perm2"] = perm[2];
        param_["perm3"] = perm[3];
        init(std::move(name), OpType::TRANSPOSE);
    }
    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class SubgraphStart final : public Layer {
public:
    SubgraphStart() = default;
    explicit SubgraphStart(const std::string &name) {
        init(name, OpType::SUBGRAPHSTART);
    }

    Tensor operator()(vector<Tensor> inputs) {
        Module::tmp_device = MLLM_QNN;
        auto ts = run(inputs, 1);
        return ts[0];
    }
};

class SubgraphFinalize final : public Layer {
public:
    SubgraphFinalize() = default;
    explicit SubgraphFinalize(const std::string &name) {
        init(name, OpType::SUBGRAPHFINALIZE);
    }

    Tensor operator()(vector<Tensor> inputs) {
        auto ts = run(inputs, 1);
        Module::tmp_device = MLLM_CPU;
        return ts[0];
    }
};

class Device2Host final : public Layer {
public:
    explicit Device2Host(const std::string &name) {
        init(name, OpType::D2H);
    }

    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class XP_KVCache final : public Layer {
public:
    explicit XP_KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = (float)n_rep;
        param_["cache_max"] = (float)cache_max;
        init(std::move(name), OpType::XP_KVCACHE);
    }

    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }
};

class ScaledDotProductAttention final : public Layer {
public:
    explicit ScaledDotProductAttention(std::string name) {
        init(std::move(name), OpType::SDPA);
    }

    // Q, K, V
    Tensor operator()(Tensor Q, Tensor K, Tensor V) {
        auto ts = run({Q, K, V}, 1); // Q, K, V
        return ts[0];
    }
};

class NTKRoPE final : public Layer {
public:
    NTKRoPE() = default;
    NTKRoPE(RoPEType type, float theta, int max_position_embeddings, int original_max_position_embeddings, const std::vector<float> &long_factor, const std::vector<float> &short_factor, std::string name) {
        init(std::move(name), OpType::NTKROPE);
        param_["pose_type"] = (float)type;
        param_["theta"] = theta;
        param_["max_position_embeddings"] = (float)max_position_embeddings;
        param_["original_max_position_embeddings"] = (float)original_max_position_embeddings;
        param_["long_factor_n"] = (float)long_factor.size();
        for (int i = 0; i < long_factor.size(); i++) {
            param_["long_factor_" + std::to_string(i)] = long_factor[i];
        }
        param_["short_factor_n"] = (float)short_factor.size();
        for (int i = 0; i < short_factor.size(); i++) {
            param_["short_factor_" + std::to_string(i)] = short_factor[i];
        }
    }

    Tensor operator()(Tensor input) {
        auto ts = run({input}, 1);
        return ts[0];
    }

    void clearCache() {
        return op_->clearCache();
    }
};

class Split final : public Layer {
public:
    Split() = default;

    explicit Split(int split_num, Chl split_dim, int split_dim_size, std::string name) {
        param_["split_num"] = (float)split_num;
        param_["split_dim"] = (float)split_dim;
        param_["split_dim_size"] = (float)split_dim_size;
        init(std::move(name), OpType::SPLIT);
    }

    vector<Tensor> operator()(Tensor input) {
        return run({input}, (int)param_["split_num"]);
    }
};

class Scale final : public Layer {
public:
    Scale() = default;

    explicit Scale(float scale, float bias, bool bias_after_scale, std::string name) {
        param_["scale"] = (float)scale;
        param_["bias"] = (float)bias;
        param_["bias_after_scale"] = (float)bias_after_scale;
        init(std::move(name), OpType::SCALE);
    }

    vector<Tensor> operator()(Tensor input) {
        return run({input}, 1);
    }
};

} // namespace mllm

#endif // OPERATION_H