//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"

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
        Module::runlistIdx = saved_list_idx;
        bool do_init = false;
        // set backend to current module device and try to create op
        backend_ = Backend::global_backends[Module::tmp_device];
        if (module->doLoad || !inited_loaded) {
            do_init = !inited_loaded;
            if (op_ == nullptr) {
                op_ = backend_->opCreate(param_, name_);
            }
            if (module->doLoad) {
                op_->load(*module->loader);
                inited_loaded = true;
            } else {
                if (!inited_loaded) {
                    module->loader = new ParamLoader("");
                    op_->load(*module->loader);
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
                if (param_["type"] == KVCACHE && do_init) {
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
        default: {
            break;
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

class RoPE final : public Layer {
public:
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
};

class IRoPE final : public Layer {
public:
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
};

class KVCache final : public Layer {
public:
    KVCache() = default;
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
    explicit RMSNorm(int norm_size, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
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
/*
class Split final : public Layer {
public:
    Split() = default;

    explicit Split(int split_num, Chl split_dim, int split_dim_size, std::string name) {
        param_["split_num"] = (float)split_num;
        param_["split_dim"] = (float)split_dim;
        param_["split_dim_size"] = (float)split_dim_size;
        init(std::move(name), OpType::SPLIT);
    }

    explicit Split(const std::vector<int> &each_dims, Chl split_dim, const std::string &name) {
        param_["split_num"] = (float)each_dims.size();
        param_["split_dim"] = (float)split_dim;
        // store each dims
        for (size_t i = 0; i < each_dims.size(); ++i) {
            param_["split_dim_size_" + std::to_string(i)] = (float)each_dims[i];
        }
        init(std::move(name), OpType::SPLIT);
    }

    vector<std::reference_wrapper<Tensor>> operator()(Tensor &input) {
        return run({input}, (int)param_["split_num"]);
    }
};
*/
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

class Concat final : public Layer {
public:
    explicit Concat(Chl axis, std::string name) {
        param_["axis"] = (float)axis;
        init(std::move(name), OpType::CAT);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        auto ts = run({input0, input1}, 1);
        return ts[0].get();
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

} // namespace mllm

#endif // OPERATION_H