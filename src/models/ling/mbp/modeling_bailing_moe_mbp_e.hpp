#pragma once
#include "DataType.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "../configuration_bailing_moe.hpp"
#include "settings_bailing_moe_mbp_e.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_NEON) && !defined(__APPLE__)
#include <pthread.h>
#include <sched.h>
#endif
#define MBP_THREAD

using namespace mllm;

class BailingMoeMLP final : public Module {
public:
    BailingMoeMLP() = default;
    BailingMoeMLP(int hidden_size, int intermediate_size, const BailingMoeNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        // 检查是否为 MoE 专家调用（需要 layer_idx 和 expert_idx）
        if (args.size() >= 2) {
            // MoE 专家模式：使用异步加载
            int layer_idx = std::any_cast<int>(args[0]);
            int expert_idx = std::any_cast<int>(args[1]);
            int next_expert_idx = args.size() > 2 ? std::any_cast<int>(args[2]) : -1;
            int next_layer_idx = args.size() > 3 ? std::any_cast<int>(args[3]) : -1;

            // 等待gate_proj加载完成
#ifdef MBP_THREAD
            {
                double wait_start_time = (mllm_time_us() - start_time) / 1000.0F;        // ms
                std::unique_lock<std::mutex> lock(*proj_mtxs[layer_idx][expert_idx][0]); // gate_proj
                proj_cvs[layer_idx][expert_idx][0]->wait(lock, [&] {
                    return proj_dones[layer_idx][expert_idx][0].load(std::memory_order_acquire) || gate_proj.loaded();
                });
                double wait_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
                std::string wait_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_gate_wait";
                expert_wait_times[wait_key] = {wait_start_time, wait_end_time};
            }
            // 计算gate_proj时异步加载up_proj (只有当前专家需要时才加载)
            if (!proj_dones[layer_idx][expert_idx][1].load(std::memory_order_acquire) && !up_proj.loaded()) {
                ProjectionLoadRequest req{layer_idx, expert_idx, 1}; // 1: up_proj
                {
                    std::lock_guard<std::mutex> lk(projection_queue_mutex);
                    projection_load_requests.push(req);
                }
                projection_queue_cv.notify_one();
            }
#endif
            assert(gate_proj.loaded() && "gate_proj should be loaded");
            double gate_start_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            auto x = gate_proj(inputs[0]);
            x = silu(x);
            double gate_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            std::string gate_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_gate";
            expert_cal_times[gate_key] = {gate_start_time, gate_end_time};

            // gate_proj计算完成后立即释放，然后请求down_proj
#ifdef MBP_THREAD
            gate_proj.free();
            if (!proj_dones[layer_idx][expert_idx][2].load(std::memory_order_acquire) && !down_proj.loaded()) {
                ProjectionLoadRequest req{layer_idx, expert_idx, 2}; // 2: down_proj
                {
                    std::lock_guard<std::mutex> lk(projection_queue_mutex);
                    projection_load_requests.push(req);
                }
                projection_queue_cv.notify_one();
            }
#endif

            // 等待up_proj加载完成并计算
#ifdef MBP_THREAD
            {
                double wait_start_time = (mllm_time_us() - start_time) / 1000.0F;        // ms
                std::unique_lock<std::mutex> lock(*proj_mtxs[layer_idx][expert_idx][1]); // up_proj
                proj_cvs[layer_idx][expert_idx][1]->wait(lock, [&] {
                    return proj_dones[layer_idx][expert_idx][1].load(std::memory_order_acquire) || up_proj.loaded();
                });
                double wait_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
                std::string wait_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_up_wait";
                expert_wait_times[wait_key] = {wait_start_time, wait_end_time};
            }
#endif
            assert(up_proj.loaded() && "up_proj should be loaded");
            double up_start_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            auto y = up_proj(inputs[0]);
            x = x * y;
            double up_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            std::string up_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_up";
            expert_cal_times[up_key] = {up_start_time, up_end_time};

            // up_proj计算完成后立即释放，然后请求下一个专家的gate_proj
#ifdef MBP_THREAD
            up_proj.free();
            // 请求下一个专家的gate_proj (只有确定需要时才预加载)
            if (next_expert_idx >= 0) {
                // 检查下一个专家的gate_proj是否需要预加载
                if (!proj_dones[layer_idx][next_expert_idx][0].load(std::memory_order_acquire)) {
                    ProjectionLoadRequest req{layer_idx, next_expert_idx, 0}; // 0: gate_proj
                    {
                        std::lock_guard<std::mutex> lk(projection_queue_mutex);
                        projection_load_requests.push(req);
                    }
                    projection_queue_cv.notify_one();
                }
            } else if (next_layer_idx >= 0) {
                // 检查下一层第一个专家的gate_proj是否需要预加载
                if (!proj_dones[next_layer_idx][0][0].load(std::memory_order_acquire)) {
                    ProjectionLoadRequest req{next_layer_idx, 0, 0}; // 下一层第一个专家的gate_proj
                    {
                        std::lock_guard<std::mutex> lk(projection_queue_mutex);
                        projection_load_requests.push(req);
                    }
                    projection_queue_cv.notify_one();
                }
            }
#endif

            // 等待down_proj加载完成并计算
#ifdef MBP_THREAD
            {
                double wait_start_time = (mllm_time_us() - start_time) / 1000.0F;        // ms
                std::unique_lock<std::mutex> lock(*proj_mtxs[layer_idx][expert_idx][2]); // down_proj
                proj_cvs[layer_idx][expert_idx][2]->wait(lock, [&] {
                    return proj_dones[layer_idx][expert_idx][2].load(std::memory_order_acquire) || down_proj.loaded();
                });
                double wait_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
                std::string wait_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_down_wait";
                expert_wait_times[wait_key] = {wait_start_time, wait_end_time};
            }
#endif
            assert(down_proj.loaded() && "down_proj should be loaded");
            double down_start_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            x = down_proj(x);
            double down_end_time = (mllm_time_us() - start_time) / 1000.0F; // ms
            std::string down_key = std::to_string(layer_idx) + "_" + std::to_string(expert_idx) + "_down";
            expert_cal_times[down_key] = {down_start_time, down_end_time};

            // down_proj计算完成后立即释放
#ifdef MBP_THREAD
            down_proj.free();
#endif
            return {x};
        } else {
            // 普通 MLP 模式：直接计算，不使用异步加载
            auto x = gate_proj(inputs[0]);
            x = silu(x);
            auto y = up_proj(inputs[0]);
            x = x * y;
            x = down_proj(x);
            return {x};
        }
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

    // 将成员变量改为公有，以便异步加载时访问
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

private:
    Layer silu;
};

class BailingMoeGate final : public Module {
public:
    BailingMoeGate() = default;
    BailingMoeGate(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const std::string &base_name) {
        gate = Linear(config.hidden_size, config.num_experts, false, base_name + "gate");
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        num_experts_per_tok = config.num_experts_per_tok;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto scores = softmax(gate(inputs[0]));
        auto experts_w_i = Tensor::topk(scores, num_experts_per_tok, DIMENSION);
        auto topk_weight = experts_w_i[0];                      //  1, batch*seq, 1, k
        auto topk_idx = experts_w_i[1];                         //  1, batch*seq, 1, k
        topk_idx = topk_idx.view(-1, 1, 1, -1);                 // 1, 1, 1, k* batch*seq
        topk_weight = topk_weight / topk_weight.sum(DIMENSION); //  1, batch*seq, 1, k
        return {scores, topk_weight, topk_idx};
    }

private:
    Layer gate;
    Softmax softmax;
    int num_experts_per_tok{};
};

class BailingMoeSparseMoeBlock final : public Module {
public:
    BailingMoeSparseMoeBlock() = default;
    BailingMoeSparseMoeBlock(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        experts = List<BailingMoeMLP>(config.num_experts, config.hidden_size, config.moe_intermediate_size, names, base_name + "experts.");
        gate = BailingMoeGate(config, names, base_name);
        num_experts_per_tok = config.num_experts_per_tok;
        num_shared_experts = config.num_shared_experts;
        num_hidden_layers = config.num_hidden_layers; // 添加层数信息
        if (num_shared_experts > 0) {
            shared_experts = BailingMoeMLP(config.hidden_size,
                                           config.moe_intermediate_size * config.num_shared_experts,
                                           names, base_name + "shared_experts.");
        }
    }

    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        auto identity = hidden_states;
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto gates_t = gate({hidden_states}); //  1, batch*seq, 1, num_experts
        auto scores = gates_t[0];             // 1, batch*seq, 1, num_experts
        auto topk_weight = gates_t[1];        // 1, batch*seq,
        auto topk_idx = gates_t[2];           // 1, batch*seq, 1, k

        // 获取层索引用于异步加载
        int layer_idx = args.size() > 0 ? std::any_cast<int>(args[0]) : 0;
        hidden_states = moe_infer(hidden_states, topk_weight, topk_idx, layer_idx); // 1, batch*seq, 1, hidden

        if (num_shared_experts) {
            hidden_states = hidden_states + shared_experts({identity})[0]; // add shared experts
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {hidden_states};
    }

    Tensor moe_infer(Tensor hidden_states, Tensor &topk_weight, Tensor &topk_idx, int layer_idx = 0) {
        auto dtype = topk_idx.dtype();
        auto device = topk_idx.device();
        topk_idx = topk_idx.fp32().cpu();
        auto idxs = topk_idx.argsort();               // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = topk_idx.bincount(); // (1, 1, 1, 0) 1, 1, 1, k
        idxs = idxs.to(device).to(dtype);
        auto token_idxs = idxs / num_experts_per_tok; // 1, 1, 1, k* batch*seq
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states); // 1, batch*seq, 1, hidden

        // 收集要处理的专家，并存储相关数据
        std::map<int, Tensor> exp_token_idx_list;
        std::map<int, Tensor> exp_idx_list;
        std::vector<int> sorted_keys;

        start_idx = 0;
        for (int i = 0; i < experts.size(); ++i) {
            if (tokens_per_expert.dimension() != 0 && i >= tokens_per_expert.dimension())
                break;
            int this_token_num = tokens_per_expert.dimension() ? tokens_per_expert.d<float>(0, 0, 0, i) : 0;
            if (!this_token_num) continue;
            end_idx = start_idx + this_token_num;
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});                         //(1, 1, 1, 0) 1, 1, 1, e-s
            if (topk_weight.dimension() != 1) { topk_weight = topk_weight.view(-1, -1, 1, 1); } // 1, k* batch*seq, 1, 1
            exp_token_idx_list[i] = exp_token_idx;
            sorted_keys.push_back(i);
            exp_idx_list[i] = exp_idx;
            start_idx = end_idx;
        }

        if (!sorted_keys.empty()) {
            // 为第一个专家预加载gate_proj
            if (!experts[sorted_keys[0]].gate_proj.loaded()) {
                double time_start = (mllm_time_us() - start_time) / 1000.0F; // ms
                experts[sorted_keys[0]].gate_proj.load();
                std::string expert_name = std::to_string(layer_idx) + "_" + std::to_string(sorted_keys[0]) + "_gate";
                double time_end = (mllm_time_us() - start_time) / 1000.0F; // ms
                proj_load_times[expert_name] = {time_start, time_end};
            }
#ifdef MBP_THREAD
            // 标记第一个专家的gate_proj为已加载
            proj_dones[layer_idx][sorted_keys[0]][0].store(true, std::memory_order_release);
            proj_cvs[layer_idx][sorted_keys[0]][0]->notify_all();
#endif
        }

        for (int ii = 0; ii < sorted_keys.size(); ii++) {
            int expert_id = sorted_keys[ii];
            if (exp_token_idx_list.find(expert_id) == exp_token_idx_list.end()) continue; // 退出
            if (Module::doLoad) continue;                                                 // 退出

            // step.1 - 准备输入数据
            double time_start_ = (mllm_time_us() - start_time) / 1000.0F; // ms

            auto exp_token_idx = exp_token_idx_list[expert_id];               //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = exp_idx_list[expert_id];                           //(1, 1, 1, 0) 1, 1, 1, e-s
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE); //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto topk_weight_clip = topk_weight.clip(exp_idx, SEQUENCE);      //(1, 0, 1, 1) 1, e-s, 1, 1

            std::string expert_name_ = std::to_string(layer_idx) + "_" + std::to_string(expert_id);
            double time_end_ = (mllm_time_us() - start_time) / 1000.0F; // ms
            expert_clip_times[expert_name_] = {time_start_, time_end_};

            auto time_start__ = (mllm_time_us());                      // ms
            double time_start = (time_start__ - start_time) / 1000.0F; // ms

            // step.2 - 执行专家计算（包含投影层级异步加载）
            // 准备下一个专家信息
            std::vector<std::any> mlp_args = {layer_idx, expert_id};
            if (ii < sorted_keys.size() - 1 && exp_token_idx_list[sorted_keys[ii + 1]].dimension() > 0) {
                mlp_args.push_back(sorted_keys[ii + 1]); // next_expert_idx
                mlp_args.push_back(-1);                  // next_layer_idx
            } else if (ii == sorted_keys.size() - 1 && layer_idx < num_hidden_layers - 1) {
                mlp_args.push_back(-1);            // next_expert_idx
                mlp_args.push_back(layer_idx + 1); // next_layer_idx
            } else {
                mlp_args.push_back(-1); // next_expert_idx
                mlp_args.push_back(-1); // next_layer_idx
            }

            auto expert_out = experts[expert_id]({expert_tokens}, mlp_args)[0]; //(1, 0, 1, hidden) 1, e-s, 1,
            expert_out = expert_out * topk_weight_clip;                         //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_add(expert_out, exp_token_idx);                // 1, batch*seq, 1, hidden
            experts[expert_id].free();

            std::string expert_name = std::to_string(layer_idx) + "_" + std::to_string(expert_id);
            auto time_end__ = (mllm_time_us());                    // ms
            double time_end = (time_end__ - start_time) / 1000.0F; // ms

#ifdef MBP_THREAD
            // 重置投影层状态
            for (int proj_type = 0; proj_type < 3; ++proj_type) {
                proj_dones[layer_idx][expert_id][proj_type].store(false, std::memory_order_relaxed);
            }
#endif
        }
        return expert_cache; // 1, batch*seq, 1, hidden
    }

    void load_experts(int expert_idx, int flag = -1) {
        switch (flag) {
        case -1: {
            experts[expert_idx].gate_proj.load();
            experts[expert_idx].up_proj.load();
            experts[expert_idx].down_proj.load();
            break;
        }
        case 0: {
            experts[expert_idx].gate_proj.load();
            break;
        }
        case 1: {
            experts[expert_idx].up_proj.load();
            break;
        }
        case 2: {
            experts[expert_idx].down_proj.load();
            break;
        }
        default:
            break;
        }
    }

private:
    BailingMoeMLP shared_experts;
    std::vector<BailingMoeMLP> experts;
    BailingMoeGate gate;
    int num_shared_experts{};
    int num_experts_per_tok{};
    int num_hidden_layers{}; // 添加层数信息
};

class BailingMoeDecoder final : public Module {
public:
    BailingMoeDecoder() = default;
    BailingMoeDecoder(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        SPLIT_HD, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta,
                                        config.max_position_embeddings,
                                        config.cache_limit, config.use_cache, config.use_qkv_bias, config.use_bias,
                                        config.attn_implementation, names, base_name + names._attn_base_name);
        moe = BailingMoeSparseMoeBlock(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        // 获取层索引，用于传递给 MoE
        int layer_idx = args.size() > 0 ? std::any_cast<int>(args[0]) : 0;

        auto hidden_states = input_layernorm(inputs[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states + inputs[0];
        hidden_states = post_attention_layernorm(tmp);

        // 传递层索引给 MoE
        std::vector<std::any> moe_args = {layer_idx};
        hidden_states = moe({hidden_states}, moe_args)[0];
        hidden_states = hidden_states + tmp;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

    void load_experts(int expert_idx, int projection_type) {
        moe.load_experts(expert_idx, projection_type);
    }

private:
    MultiHeadAttention self_atten;
    BailingMoeSparseMoeBlock moe;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    int num_hidden_layers;
};

class BailingMoeModel final : public Module {
public:
    BailingMoeModel() = default;
    BailingMoeModel(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        blocks = List<BailingMoeDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        for (int i = 0; i < blocks.size(); ++i) {
            // 传递层索引给每个decoder block
            std::vector<std::any> block_args = {i};
            hidden_states = blocks[i]({hidden_states}, block_args)[0];
        }
        hidden_states = norm(hidden_states);
        return {hidden_states};
    }

    void load_experts(int layer_idx, int expert_idx, int projection_type) {
        if (layer_idx >= 0 && layer_idx < blocks.size()) {
            blocks[layer_idx].load_experts(expert_idx, projection_type);
        }
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
    std::vector<BailingMoeDecoder> blocks;
    Layer norm;
};

class BailingMoeForCausalLM final : public Module {
public:
    CHAINABLE_MODULE_METHODS(BailingMoeForCausalLM)
    BailingMoeForCausalLM(BailingMoeConfig &config) {
        dtype = config.dtype;
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = BailingMoeModel(config, names, names.blk_name);
        lm_head = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);

        // 初始化异步加载相关设置
        num_layers = config.num_hidden_layers;
        num_experts = config.num_experts;
        ling_mbp_init(num_layers, num_experts);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        clearMBPtimes();
        start_time = mllm_time_us();

        auto x = embedding(inputs[0]).to(dtype);
        std::vector<std::any> empty_args; // 为 model 创建空的参数
        auto outputs = model({x}, empty_args)[0];
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
        outputs = lm_head(outputs);
        return {outputs};
    }

    void load_projection(int layer_idx, int expert_idx, int projection_type) {
        switch (projection_type) {
        case 0: // gate_proj
            model.load_experts(layer_idx, expert_idx, 0);
            break;
        case 1: // up_proj
            model.load_experts(layer_idx, expert_idx, 1);
            break;
        case 2: // down_proj
            model.load_experts(layer_idx, expert_idx, 2);
            break;
        default:
            model.load_experts(layer_idx, expert_idx, -1);
            break;
        }
    }

    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Layer lm_head;
    BailingMoeModel model;
    DataType dtype;
    int num_layers{};
    int num_experts{};
};
