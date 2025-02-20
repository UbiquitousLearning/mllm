//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

#include "OpDefined.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Trace.hpp"

#include <Module.hpp>

#include <regex>
#include <string>
#include <vector>

namespace mllm {

class Layer {
public:
    Layer() = default;
    void init(std::string name, OpType type) {
        name_ = std::move(name);
        param_["type"] = type;
        Module::initBackend(MLLM_CPU);
        backend_ = Backend::global_backends[MLLM_CPU];
        saved_list_idx = Module::listIdx;
        init_ = true;
    }
    bool ready() {
        return init_;
    }
    bool inited_loaded = false;
    bool loaded_param = false;
    static map<string, string> layername_2_tensorname;
    static bool use_layername_2_tensorname;

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }

    Tensor &operator()(Tensor &input0, Tensor &input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0].get();
    }

    Tensor &operator()(Tensor &input0, Tensor &input1, Tensor &input2) {
        auto ts = run({input0, input1, input2}, 1);
        return ts[0].get();
    }

    Tensor &operator()(Tensor &input0, Tensor &input1, Tensor &input2, Tensor &input3) {
        auto ts = run({input0, input1, input2, input3}, 1);
        return ts[0].get();
    }

    void load() {
        if (inited_loaded && loaded_param)
            return;
        if (op_ == nullptr) {
#ifdef USE_QNN
            if ((param_["type"] == KVCACHE || param_["type"] == KVCACHENPU) && (Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end())) {
                if (kv_cache_map.find(name_) == kv_cache_map.end()) {
                    // for the prefill part, we need to create a new op
                    param_["type"] = KVCACHENPU;
                    op_ = backend_->opCreate(param_, name_);
                    kv_cache_map[name_] = op_;
                } else {
#ifdef DEBUGPRINT
                    std::cout << name_ << " is shared used" << std::endl;
#endif
                    // for the decoding part, we need to get created op from global container
                    op_ = kv_cache_map[name_];
                }
            } else {
                op_ = backend_->opCreate(param_, name_);
            }
#else
            op_ = backend_->opCreate(param_, name_);
#endif
        }
        op_->load(*Module::llm_model_ptr->loader);
        loaded_param = true;
    }
    bool &loaded() {
        return loaded_param;
    }
    void free() {
        op_->free({}, {});
        loaded_param = false;
    }

private:
    std::string name_num_to_X(const std::string &input_string) {
        std::regex pattern(R"(\.\d{1,3}\.)"); // Matches any number between 1 and 100 between two dots
        std::string replacement = ".X.";      // The string to replace the matched pattern with
        std::string output_string = std::regex_replace(input_string, pattern, replacement);
        return output_string;
    }
    std::string name_X_to_num(const std::string &input_string, int in_idx) {
        std::regex pattern(".X.");                                    // Matches any number between 1 and 100 between two dots
        std::string replacement = "." + std::to_string(in_idx) + "."; // The string to replace the matched pattern with
        std::string output_string = std::regex_replace(input_string, pattern, replacement);
        return output_string;
    }
    void init_reset_KVCache(string input_name, Module *module) {
        map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
        vector<string> renameX_names;
        renameX_names.push_back(input_name);
        const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2", "-cat", "-split-0-48"};
        vector<string> new_names;
        bool can_break = true;
        auto in_x_name = renameX_names[0];
        while (can_break) {
            can_break = false;
            for (const auto &suffix : suffixs) {
                if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                    const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                    if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end() && std::find(new_names.begin(), new_names.end(), r_name) == new_names.end()) {
                        new_names.push_back(r_name);
                        in_x_name = r_name;
                        can_break = true;
                    }
                    break;
                }
            }
        }
        renameX_names.insert(renameX_names.end(), new_names.begin(), new_names.end());
        for (const auto x_name : renameX_names) {
            auto name = name_X_to_num(x_name, saved_list_idx);
            layername_2_tensorname[name] = name;
            activation_tensors[name] = std::make_shared<Tensor>(backend_);
            activation_tensors[name]->initFrom(*activation_tensors[x_name]);
            activation_tensors[name]->setName(name);
            activation_tensors[name]->setModule(module);
        }
    }

protected:
    vector<std::reference_wrapper<Tensor>> run(vector<Tensor> inputs, int N = 1) {
        Module *module;
        if (!inputs.empty()) {
            module = inputs[0].module();
        } else {
            module = Module::llm_model_ptr;
        }
        map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
        auto &activation_tensors_num = module->activation_tensors_num;
        // Module::runlistIdx = saved_list_idx;
        bool do_init = false;

        if (module->doLoad || !inited_loaded) {
            // set backend to current module device and try to create op
            // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
            backend_ = Backend::global_backends[Module::tmp_device];
            do_init = !inited_loaded;
            if (op_ == nullptr) {
#ifdef USE_QNN
                if ((param_["type"] == KVCACHE || param_["type"] == KVCACHENPU) && (Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end())) {
                    if (kv_cache_map.find(name_) == kv_cache_map.end()) {
                        // for the prefill part, we need to create a new op
                        param_["type"] = KVCACHENPU;
                        op_ = backend_->opCreate(param_, name_);
                        kv_cache_map[name_] = op_;
                    } else {
#ifdef DEBUGPRINT
                        std::cout << name_ << " is shared used" << std::endl;
#endif
                        // for the decoding part, we need to get created op from global container
                        op_ = kv_cache_map[name_];
                    }
                } else {
                    op_ = backend_->opCreate(param_, name_);
                }
#else
                op_ = backend_->opCreate(param_, name_);
#endif
            }
            if (module->doLoad) {
                op_->load(*module->loader);
                inited_loaded = true;
            } else if (loaded_param) {
                inited_loaded = loaded_param;
            } else {
                if (!inited_loaded) {
                    // module->loader = new ParamLoader("");
                    // op_->load(*module->loader);
                    auto empty_loader = new ParamLoader("");
                    op_->load(*empty_loader);
                    inited_loaded = true;
                }
            }
            vector<string> layer_next_names = {};
            if (N > 1) {
                for (int i = 0; i < N; ++i) {
                    layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
                }
            } else {
                layer_next_names = {"out-" + op_->name()};
            }
            for (const auto &layer_next_name : layer_next_names) {
                string next_name;
                if (use_layername_2_tensorname) {
                    if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                        if (param_["type"] == KVCACHE) {
                            layername_2_tensorname[layer_next_name] = layer_next_name;
                            init_reset_KVCache(inputs[0].name(), module);
                        } else {
                            layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                        }
                    }
                    next_name = layername_2_tensorname[layer_next_name];
                } else {
                    next_name = layer_next_name;
                }
                if (activation_tensors.find(next_name) == activation_tensors.end()) {
                    activation_tensors[next_name] = std::make_shared<Tensor>(backend_);
                    activation_tensors[next_name]->setName(next_name);
                    activation_tensors[next_name]->setModule(module);
                    activation_tensors_num[next_name] = 0;
                }
            }
            if (module->doLoad) {
                vector<std::reference_wrapper<Tensor>> output_result = {};
                for (const auto &layer_next_name : layer_next_names) {
                    string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
                    output_result.push_back(*activation_tensors[next_name]);
                }
                return output_result;
            }
        }
        // input_tensors
        vector<shared_ptr<Tensor>> input_tensors;
        for (auto &input : inputs) {
            if (input.shouldInGraphs()) {
                auto input_name = input.name();
                if (param_["type"] == KVCACHE && do_init && use_layername_2_tensorname) {
                    input_name = name_X_to_num(input_name, saved_list_idx);
                }
                input_tensors.push_back(activation_tensors[input_name]);
            } else {
                input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
            }
        }
        // output_tensors
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + op_->name()};
        }
        vector<shared_ptr<Tensor>> output_tensors = {};
        for (const auto &layer_next_name : layer_next_names) {
            string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
            output_tensors.push_back(activation_tensors[next_name]);
        }
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
        case TENSOR_STATIC_INIT: {
            op_->reshape(input_tensors, output_tensors);
            op_->setUp(input_tensors, output_tensors);
            break;
        }
        case TENSOR_STATIC_READY: {
            op_->execute(input_tensors, output_tensors);
            break;
        }
        case TENSOR_STATIC_TRACE : {
            if (backend_->type() == BackendType::MLLM_CPU) {
                Tracer::addOp(op_, input_tensors, output_tensors);
            }
            break;
        }
        default: {
            break;
        }
        }
        if (Backend::global_backends.size() == 1) {
            for (auto input_tensor : input_tensors) {
                if ((activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end())) {
                    switch (Tensor::tensor_status) {
                    case TENSOR_STATIC_INIT: {
                        activation_tensors_num[input_tensor->name()] += 1;
                        break;
                    }
                    case TENSOR_STATIC_READY: {
                        activation_tensors_num[input_tensor->name()] -= 1;
                        break;
                    }
                    default: {
                    }
                    }
                    if (activation_tensors_num[input_tensor->name()] == 0 && activation_tensors[input_tensor->name()]->sequence() > 1
                        && activation_tensors[input_tensor->name()]->ttype()!= GRAPH_OUTPUT) {
                        activation_tensors[input_tensor->name()]->free();
                        // std::cout << input_tensor->name() << "|" << std::endl;
                    }
                }
            }
        }
#ifdef DEBUGOPTIME
        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
            auto end_t = mllm_time_us();
            std::cout << op_->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
        }
#endif
        vector<std::reference_wrapper<Tensor>> output_result = {};
        for (const auto &layer_next_name : layer_next_names) {
            string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
#ifdef DEBUGSAVETENSOR
            activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
            output_result.push_back(*activation_tensors[next_name]);
        }
        return output_result;
    }

    std::string name_;
    Op *op_ = nullptr;
    Backend *backend_{};
    OpParam param_;
    bool init_ = false;
    int saved_list_idx;
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
        auto ts = run({input}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class SparseIdLinear final : public Layer {
public:
    SparseIdLinear(int in_dim, int out_dim, std::string name) {
        param_["in_dim_"] = (float)in_dim;
        param_["out_dim_"] = (float)out_dim;
        init(std::move(name), OpType::SPARSEIDLINEAR);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class SparseLinear final : public Layer {
public:
    SparseLinear(int in_dim, int out_dim, std::string name) {
        param_["in_dim_"] = (float)in_dim;
        param_["out_dim_"] = (float)out_dim;
        init(std::move(name), OpType::SPARSELINEAR);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Predictor final : public Layer {
public:
    Predictor(int in_dim, int out_dim, std::string name) {
        param_["in_dim"] = (float)in_dim;
        param_["out_dim"] = (float)out_dim;
        init(std::move(name), OpType::PREDICTOR);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input0, int activate_input_dim, int activate_output_dim) {
        auto activate_input_dim_tensor = Tensor(activate_input_dim, backend_);
        auto activate_output_dim_tensor = Tensor(activate_output_dim, backend_);
        auto ts = run({input0, activate_input_dim_tensor, activate_output_dim_tensor}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input0, Tensor &input1, Tensor &input2) {
        auto ts = run({input0, input1, input2}, 1);
        return ts[0].get();
    }
};

class SiLU final : public Layer {
public:
    SiLU() = default;
    SiLU(std::string name) {
        init(std::move(name), OpType::SILU);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class ReLU final : public Layer {
public:
    ReLU() = default;
    ReLU(std::string name) {
        init(std::move(name), OpType::RELU);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class ReLUSquaredActivation final : public Layer {
public:
    ReLUSquaredActivation() = default;
    ReLUSquaredActivation(std::string name) {
        init(std::move(name), OpType::RELU2);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class GELU final : public Layer {
public:
    GELU() = default;
    GELU(std::string name) {
        init(std::move(name), OpType::OP_GELU);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class QuickGELU final : public Layer {
public:
    QuickGELU() = default;
    explicit QuickGELU(std::string name) {
        init(std::move(name), OpType::QUICKGLUE);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
    Tensor &operator()(Tensor &input, int axis_classes) {
        auto axis_classes_tensor = Tensor(axis_classes, backend_);
        auto ts = run({input, axis_classes_tensor}, 1);
        return ts[0].get();
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
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Causalmask final : public Layer {
public:
    Causalmask() = default;
    explicit Causalmask(std::string name) {
        init(std::move(name), OpType::CAUSALMASK);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
    Tensor &operator()(Tensor &input0, int kvcache_seq) {
        auto kvcache_seq_tensor = Tensor(kvcache_seq, backend_);
        auto ts = run({input0, kvcache_seq_tensor}, 1);
        return ts[0].get();
    }
};

class SlidingWindowMask final : public Layer {
public:
    explicit SlidingWindowMask(int window_size, std::string name) {
        param_["window_size"] = window_size;
        init(std::move(name), OpType::SLIDINGWINDOWMASK);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

typedef std::unordered_map<string, std::any> RoPEConfig;

class RoPE final : public Layer {
public:
    RoPE() = default;

    explicit RoPE(int pose_type, const RoPEConfig & config, std::string name) {
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
    void clearCache() {
        return op_->clearCache();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
    void clearCache() {
        return op_->clearCache();
    }
};

class KVCache final : public Layer {
public:
    KVCache() = default;
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
    int getCacheSeqLen() {
        return op_->getCacheSeqLen();
    }
    void clearCache() {
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Matmul final : public Layer {
public:
    explicit Matmul(bool transpose0, bool transpose1, std::string name) {
        param_["transpose0"] = transpose0;
        param_["transpose1"] = transpose1;
        init(std::move(name), OpType::MATMUL);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class VisionRoPE final : public Layer {
public:
    explicit VisionRoPE(int dim_size, int spatial_merge_size, std::string name) {
        param_["dim"] = (float)dim_size;
        param_["spatial_merge_size"] = (float)spatial_merge_size;
        init(std::move(name), OpType::VISIONROPE);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
        init(std::move(name), OpType::MULTIMODALROPE);
    }
    Tensor &operator()(Tensor &input, Tensor &position_ids) {
        auto ts = run({input, position_ids}, 1);
        return ts[0].get();
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
    Tensor &operator()() {
        auto ts = run({}, 1);
        return ts[0].get();
    }
};

class Position final : public Layer {
public:
    explicit Position(std::string name) {
        init(std::move(name), OpType::POSITION);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};












//  Only for QNN START

class Quantize final : public Layer {
public:
    explicit Quantize(bool isNSHD, std::string name) {
        param_["isNSHD"] = (float)isNSHD;
        init(std::move(name), OpType::QUANTIZE);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
    explicit Dequantize(bool isNSHD, std::string name, bool isFP32 = true) {
        param_["isNSHD"] = (float)isNSHD;
        param_["isFP32"] = (float)isFP32;
        init(std::move(name), OpType::DEQUANTIZE);
    }
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Dispatch final : public Layer {
public:
    explicit Dispatch(const std::string &name) {
        init(name, OpType::DISPATCH);
    }

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Add final : public Layer {
public:
    explicit Add(std::string name) {
        init(std::move(name), OpType::ADD);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0].get();
    }
};

class Mul final : public Layer {
public:
    explicit Mul(std::string name) {
        init(std::move(name), OpType::MUL);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class SubgraphStart final : public Layer {
public:
    explicit SubgraphStart(const std::string &name) {
        init(name, OpType::SUBGRAPHSTART);
    }

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
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
    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class SubgraphFinalize final : public Layer {
public:
    explicit SubgraphFinalize(const std::string &name) {
        init(name, OpType::SUBGRAPHFINALIZE);
    }

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class Device2Host final : public Layer {
public:
    explicit Device2Host(const std::string &name) {
        init(name, OpType::D2H);
    }

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class XP_KVCache final : public Layer {
public:
    explicit XP_KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = (float)n_rep;
        param_["cache_max"] = (float)cache_max;
        init(std::move(name), OpType::XP_KVCACHE);
    }

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};

class ScaledDotProductAttention final : public Layer {
public:
    explicit ScaledDotProductAttention(std::string name) {
        init(std::move(name), OpType::SDPA);
    }

    // Q, K, V
    Tensor &operator()(Tensor &Q, Tensor &K, Tensor &V) {
        auto ts = run({Q, K, V}, 1); // Q, K, V
        return ts[0].get();
    }
};

class NTKRoPE final : public Layer {
public:
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

    Tensor &operator()(Tensor &input) {
        auto ts = run({input}, 1);
        return ts[0].get();
    }
};
//  Only for QNN END

} // namespace mllm

#endif // OPERATION_H