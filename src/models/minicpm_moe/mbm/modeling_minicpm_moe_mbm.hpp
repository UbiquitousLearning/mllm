#ifndef MODELING_MINICPMMOE_MBM_HPP
#define MODELING_MINICPMMOE_MBM_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
// #include "backends/cpu/CPUBackend.hpp"
#include "../configuration_minicpm_moe.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cassert>
#include <cmath>
// #include <mutex>
#include <condition_variable>
// #include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include "settings_minicpm_moe_mbm.hpp"

// #define MTIME

#ifdef MTIME
#include "Timing.hpp"
#include <iostream>
#endif

using namespace mllm;

class MiniCPMMLP final : public Module {
public:
    MiniCPMMLP() = default;
    MiniCPMMLP(int hidden_size, int intermediate_size, const MiniCPMNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]); // ERROR
        x = x * y;
        x = down_proj(x);
        return {x};
    }

    void load() {
        gate_proj.load();
        up_proj.load();
        down_proj.load();
    }
    bool loaded() {
        return gate_proj.loaded() && up_proj.loaded() && down_proj.loaded();
    }

    void free() {
        gate_proj.free();
        up_proj.free();
        down_proj.free();
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer silu;
};

#ifdef MTIME
int64_t end_infer_last = 0;
#endif
int mbm_idxs_size;
class MiniCPMMoE final : public Module {
public:
    MiniCPMMoE() = default;
    MiniCPMMoE(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        layer_idx = Module::listIdx;
        experts = List<MiniCPMMLP>(config.num_experts, config.hidden_size, config.intermediate_size, names, base_name + "experts.");
        gate = Linear(config.hidden_size, config.num_experts, false, base_name + "gate");
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        num_experts_per_tok = config.num_experts_per_tok;
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto scores = gate(hidden_states); //  1, batch*seq, 1, num_experts
        scores = softmax(scores);
        auto experts_w_i = Tensor::topk(scores, num_experts_per_tok, DIMENSION);
        auto expert_weights = experts_w_i[0].get();                      //  1, batch*seq, 1, k
        auto expert_indices = experts_w_i[1].get();                      //  1, batch*seq, 1, k
        expert_indices = expert_indices.view(-1, 1, 1, -1);              // 1, 1, 1, k* batch*seq
        expert_weights = expert_weights / expert_weights.sum(DIMENSION); //  1, batch*seq, 1, k
        expert_weights = expert_weights.view(-1, -1, 1, 1);              // 1, k* batch*seq, 1, 1
        auto idxs = expert_indices.argsort();                            // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = expert_indices.bincount();              // (1, 1, 1, 0) 1, 1, 1, k
        /*
        load_experts_1th(tokens_per_expert);
        auto expert_cache = moe_infer(hidden_states, tokens_per_expert, expert_weights, idxs);
        */
        Tensor expert_cache;
#ifdef MTIME
        if (Tensor::tensor_status == TENSOR_STATIC_READY && hidden_states.sequence() == 1) {
            std::cout << "attn  || exe time: " << (mllm_time_us() - end_infer_last) / 1000.0F << "ms" << std::endl;
        }
#endif
        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
            vector<int> tokens_per_expert_vector;
            for (int i = 0; i < tokens_per_expert.dimension(); ++i) {
                if (tokens_per_expert.d<float>(0, 0, 0, i)) {
                    tokens_per_expert_vector.push_back(i);
                }
            }
            std::cout << "";
            //
            if (layer_idx < 39 && tokens_per_expert_vector.size() == 2) {
                if (mbm_maps[layer_idx].find(tokens_per_expert_vector) != mbm_maps[layer_idx].end()) {
                    mbm_load_expert_idxs.clear();
                    auto c = mbm_maps[layer_idx][tokens_per_expert_vector];
                    mbm_load_expert_idxs = c[0];
                    mbm_load_layer_idx = layer_idx + 1;
                    do_mbm_load = true;
                }
            } else if (layer_idx == 39 && tokens_per_expert_vector.size() == 2) {
                if (mbm_maps[layer_idx].find(tokens_per_expert_vector) != mbm_maps[layer_idx].end()) {
                    mbm_load_expert_idxs.clear();
                    auto c = mbm_maps[layer_idx][tokens_per_expert_vector];
                    mbm_load_expert_idxs = c[0];
                    mbm_load_layer_idx = 0;
                    do_mbm_load = true;
                }
            }
            /*
            mbm_load_expert_idxs = mbm_idxs;
            mbm_load_layer_idx = layer_idx;
            do_mbm_load = true;
            */
            if (mbm_idxs_size == 2 && tokens_per_expert_vector.size() == 2) {     // layer_idx > 0 && && layer_idx < 39
                int &done = dones[layer_idx];                                     // 标志变量，用于表示数据是否已被修改
                cvs[layer_idx]->wait(locks[layer_idx], [&done] { return done; }); // 等待条件满足
                assert(dones[layer_idx]);
            }
            if (!experts_loaded(tokens_per_expert_vector)) {
                load_experts(tokens_per_expert_vector);
            }
            assert(experts_loaded(tokens_per_expert_vector));
            expert_cache = moe_infer(hidden_states, tokens_per_expert, expert_weights, idxs);
            if (mbm_idxs_size == 2 && tokens_per_expert_vector.size() == 2) { // layer_idx > 0 &&  && layer_idx < 39
                reset_syntax_mbm(layer_idx);
            }
            if (layer_idx == 0)
                mbm_idxs_size = tokens_per_expert_vector.size();
        } else {
            expert_cache = moe_infer(hidden_states, tokens_per_expert, expert_weights, idxs);
        }
#ifdef MTIME
        if (Tensor::tensor_status == TENSOR_STATIC_READY && hidden_states.sequence() == 1) {
            end_infer_last = mllm_time_us();
        }
#endif
        return {expert_cache};
    }

    void load_experts(vector<int> expert_idxs) {
        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
#ifdef MTIME
            auto start_infer = mllm_time_us();
#endif
            int result;
            // #pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int i = 0; i < expert_idxs.size(); ++i) {
                if (expert_idxs.size() == 2) {
                    if (std::find(mbm_v[layer_idx].begin(), mbm_v[layer_idx].end(), expert_idxs[i]) != mbm_v[layer_idx].end()) {
                        // 在 mbm_v[layer_idx] 中找到了 expert_idxs[i]
                        if (experts[expert_idxs[i]].loaded()) {
                            continue;
                        } else {
                            std::cout << "[ERROR] experts load." << std::endl;
                            experts[expert_idxs[i]].load();
                            continue;
                        }
                    }
                    if (mbm_v[layer_idx].size() >= mbm_num_max_experts) {
                        result = mbm_queue_remove(mbm_v[layer_idx], expert_idxs);
                        if (result != -1) { // mbm_v[layer_idx]不全是expert_idxs
                            experts[result].free();
                            mbm_v[layer_idx].push_back(expert_idxs[i]);
                            // if (mbm_load_layer_idx != layer_idx)
                            //     std::cout << layer_idx << " " << mbm_load_layer_idx << "  : " << expert_idxs[i] << std::endl;
                            experts[expert_idxs[i]].load();
                        }
                    } else {
                        mbm_v[layer_idx].push_back(expert_idxs[i]);
                        experts[expert_idxs[i]].load();
                    }
                    assert(experts[expert_idxs[i]].loaded());
                } else {
                    experts[expert_idxs[i]].load();
                }
            }
#ifdef MTIME
            if (expert_idxs.size() == 2) {
                auto end_infer = mllm_time_us();
                std::cout << "expert|| load time: " << (end_infer - start_infer) / 1000.0F << "ms" << std::endl;
            }
#endif
        }
    }

private:
    bool experts_loaded(vector<int> expert_idxs) {
        for (int i = 0; i < expert_idxs.size(); ++i) {
            if (!experts[expert_idxs[i]].loaded()) {
                return false;
            }
        }
        return true;
    }
    void load_experts_1th(Tensor &tokens_per_expert) {
        if (tokens_per_expert.dimension()) {
            vector<int> expert_idxs;
            for (int i = 0; i < tokens_per_expert.dimension(); ++i) {
                if (tokens_per_expert.d<float>(0, 0, 0, i)) {
                    expert_idxs.push_back(i);
                }
            }
            // load_experts(expert_idxs);
            for (int i = 0; i < expert_idxs.size(); ++i) {
                experts[expert_idxs[i]].load();
            }
        }
    }
    void load_experts(Tensor &tokens_per_expert) {
        if (tokens_per_expert.dimension()) {
            vector<int> expert_idxs;
            for (int i = 0; i < tokens_per_expert.dimension(); ++i) {
                if (tokens_per_expert.d<float>(0, 0, 0, i)) {
                    expert_idxs.push_back(i);
                }
            }
            load_experts(expert_idxs);
        }
    }
    void free_experts(vector<int> expert_idxs) {
        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
            for (int i = 0; i < expert_idxs.size(); ++i) {
                experts[expert_idxs[i]].free();
            }
        }
    }
    Tensor moe_infer(Tensor &hidden_states, Tensor &tokens_per_expert, Tensor &expert_weights, Tensor &idxs) {
#ifdef MTIME
        auto start_infer = mllm_time_us();
#endif
        auto token_idxs = idxs / num_experts_per_tok; // 1, 1, 1, k* batch*seq
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states); // 1, batch*seq, 1, hidden
        for (int i = 0; i < experts.size(); ++i) {
            if (Module::llm_model_ptr->doLoad || (tokens_per_expert.dimension() != 0 && i >= tokens_per_expert.dimension())) {
                break;
            }
            int this_token_num = tokens_per_expert.dimension() == 0 ?
                                     0 :
                                     tokens_per_expert.d<float>(0, 0, 0, i);
            if (tokens_per_expert.dimension() != 0 && this_token_num == 0)
                continue;
            end_idx = start_idx + this_token_num;
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx}); //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE);       //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto expert_out = experts[i]({expert_tokens})[0];                       //(1, 0, 1, hidden) 1, e-s, 1,

            if (hidden_states.sequence() != 1 && tokens_per_expert.dimension()) {
                free_experts({i});
            }

            auto expert_weights_clip = expert_weights.clip(exp_idx, SEQUENCE); //(1, 0, 1, 1) 1, e-s, 1, 1
            expert_out = expert_out * expert_weights_clip;                     //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_reduce(expert_out, exp_token_idx);            // 1, batch*seq, 1, hidden
            start_idx = end_idx;
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);
        }
#ifdef MTIME
        if (Tensor::tensor_status == TENSOR_STATIC_READY && hidden_states.sequence() == 1) {
            auto end_infer = mllm_time_us();
            std::cout << "expert|| exe time: " << (end_infer - start_infer) / 1000.0F << "ms" << std::endl;
        }
#endif
        return expert_cache;
    }

    std::vector<MiniCPMMLP> experts;
    Layer gate;
    Softmax softmax;
    int num_experts_per_tok{};
    int layer_idx{};
};

class MiniCPMDecoder final : public Module {
public:
    MiniCPMDecoder() = default;
    MiniCPMDecoder(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, names, base_name + names._attn_base_name);
        moe = MiniCPMMoE(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        scale_depth = config.scale_depth;
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = input_layernorm(inputs[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + inputs[0];
        hidden_states = post_attention_layernorm(tmp);
        hidden_states = moe({hidden_states})[0];
        hidden_states = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + tmp;
        return {hidden_states};
    }
    void load_experts(vector<int> expert_idxs) {
        moe.load_experts(expert_idxs);
    }
    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    MiniCPMMoE moe;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    float scale_depth;
    int num_hidden_layers;
};

class MiniCPMModel final : public Module {
public:
    MiniCPMModel() = default;
    MiniCPMModel(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        blocks = List<MiniCPMDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }
    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        for (auto &block : blocks) {
            hidden_states = block({hidden_states})[0];
        }
        hidden_states = norm(hidden_states);
        return {hidden_states};
    }
    void load_experts(int layer_idx, vector<int> expert_idxs) {
        blocks[layer_idx].load_experts(expert_idxs);
    }
    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }

private:
    std::vector<MiniCPMDecoder> blocks;
    Layer norm;
};

class MiniCPMForCausalLM final : public Module {
public:
    MiniCPMForCausalLM(MiniCPMConfig &config) {
        KVCache_TYPE = 32;
        auto names = config.names_config;
        scale_emb = config.scale_emb;
        dim_model_base = config.dim_model_base;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = MiniCPMModel(config, names, names.blk_name);
        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        std::vector<Tensor> outputs;
        if (Tensor::tensor_status == TENSOR_STATIC_READY && inputs[0].sequence() == 1) {
            omp_set_max_active_levels(2); // Enable OpenMP nesting
#pragma omp parallel num_threads(2)
            if (omp_get_thread_num() == 0) { // 根据线程ID决定执行哪个函数
#if defined(__ARM_NEON)
                // 绑定线程到特定的CPU核心
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(6, &cpuset); // 假设小核心是CPU 6
                pid_t current_thread = gettid();
                sched_setaffinity(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
                mbm_load();
            } else {
                outputs = do_forward(inputs, args);
            }
        } else {
            outputs = do_forward(inputs, args);
        }
        return outputs;
    }
    std::vector<Tensor> do_forward(std::vector<Tensor> inputs, std::vector<std::any> args) {
        auto x = embedding(inputs[0]) * scale_emb;
        auto outputs = model({x})[0];
        outputs = outputs / (hidden_size / dim_model_base);
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        mbm_finish = false;
        return {outputs};
    }
    void load_experts(int layer_idx, vector<int> expert_idxs) {
        model.load_experts(layer_idx, expert_idxs);
    }
    void clear_kvcache() override {
        model.clear_kvcache();
    }

    void mbm_load() {
        mbm_finish = true;
        while (mbm_finish) {
#if defined(__ARM_NEON)
            // sched_yield();
            std::atomic_thread_fence(std::memory_order_acquire); // 确保内存可见性
#endif
            if (do_mbm_load) {
                load_experts(mbm_load_layer_idx, mbm_load_expert_idxs);
                do_mbm_load = false;
                // 同步
                dones[mbm_load_layer_idx] = true;      // 设置标志，表示数据已被修改
                cvs[mbm_load_layer_idx]->notify_one(); // 通知等待的打印线程，数据已经修改好了，可以打印了
                locks[mbm_load_layer_idx].unlock();    // 释放互斥锁
            }
        }
    }

private:
    int hidden_size;
    float dim_model_base;
    bool tie_embedding_words;
    float scale_emb;
    Layer embedding;
    Parameter lm_head;
    MiniCPMModel model;
};

#endif // MODELING_MINICPMMOE_MBM_HPP
