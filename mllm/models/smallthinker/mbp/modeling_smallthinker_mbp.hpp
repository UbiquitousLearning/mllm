#ifndef MODELING_SMOLTHINKER_HPP
#define MODELING_SMOLTHINKER_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../configuration_smallthinker.hpp"
#include "settings_smallthinker_mbp.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <algorithm>
#include <any>
#include <set>
#include <vector>
// #include <iostream>
#include <omp.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <map>
#include <string>
#include <memory>
#if defined(__ARM_NEON) && !defined(__APPLE__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

#define MBP_THREAD

using namespace mllm;

class SmallThinkerMLP final : public Module {
public:
    SmallThinkerMLP() = default;
    SmallThinkerMLP(int hidden_size, int intermediate_size, const SmallThinkerNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        relu = ReLU(base_name + "relu");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = relu(x);
        auto y = up_proj(inputs[0]);
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
    Layer relu;
};

class SmallThinkerMoeBlock final : public Module {
public:
    SmallThinkerMoeBlock() = default;
    SmallThinkerMoeBlock(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        experts = List<SmallThinkerMLP>(config.num_experts, config.hidden_size, config.intermediate_size, names, base_name + "experts.");
        // primary_router = Linear(config.hidden_size, config.num_experts, false, base_name + "primary_router");
        sigmoid = Sigmoid(base_name + "sigmoid");
        num_experts_per_tok = config.num_experts_per_tok;
        num_hidden_layers = config.num_hidden_layers;
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        int layer_idx = std::any_cast<int>(args[0]);
        if (hidden_states.batch() > 1) hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        auto router_logits = inputs[1];
        auto expert_indices = inputs[2];
        auto expert_weights = sigmoid(router_logits);
        expert_weights = expert_weights / expert_weights.sum(DIMENSION); //  1, batch*seq, 1, k
        expert_weights = expert_weights.view(-1, -1, 1, 1);              // 1, k* batch*seq, 1, 1
        // moe_infer
        auto idxs = expert_indices.argsort();               // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = expert_indices.bincount(); // (1, 1, 1, 0) 1, 1, 1, k
        auto token_idxs = idxs / num_experts_per_tok;       // 1, 1, 1, k* batch*seq
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states); // 1, batch*seq, 1, hidden
        for (int i = 0; i < experts.size(); ++i) {
            if (Module::llm_model_ptr->doTrace || (tokens_per_expert.dimension() != 0 && i >= tokens_per_expert.dimension())) {
                break;
            }
            int this_token_num = tokens_per_expert.dimension() == 0 ?
                                     0 :
                                     tokens_per_expert.d<float>(0, 0, 0, i);
            if (tokens_per_expert.dimension() != 0 && this_token_num == 0)
                continue;
            end_idx = start_idx + this_token_num;
            //
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx}); //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s

            // step.1 - 裁剪数据
            double time_start_ = (mllm_time_us() - start_time) / 1000.0F;      // ms
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE);  //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto expert_weights_clip = expert_weights.clip(exp_idx, SEQUENCE); //(1, 0, 1, 1) 1, e-s, 1, 1

            string expert_name_ = std::to_string(layer_idx) + "_" + std::to_string(i);
            double time_end_ = (mllm_time_us() - start_time) / 1000.0F; // ms
            expert_clip_times[expert_name_] = {time_start_, time_end_};

#ifdef MBP_THREAD
            // step.2 - 等待加载完成
            double time_start_w = (mllm_time_us() - start_time) / 1000.0F; // ms
            if (!experts[i].loaded()) {
                unique_lock<mutex> lock(*mtxs[layer_idx][i]); // 局部锁
                cvs[layer_idx][i]->wait(lock, [&] {
                    return dones[layer_idx][i].load(memory_order_acquire);
                });
                assert(dones[layer_idx][i]);
            }
            double time_end_w = (mllm_time_us() - start_time) / 1000.0F; // ms
            expert_wait_times[expert_name_] = {time_start_w, time_end_w};
#endif
            auto time_start__ = (mllm_time_us());                      // ms
            double time_start = (time_start__ - start_time) / 1000.0F; // ms

            // step.3 - 专家计算
            auto expert_out = experts[i]({expert_tokens})[0];    //(1, 0, 1, hidden) 1, e-s, 1,
            expert_out = expert_out * expert_weights_clip;       //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_add(expert_out, exp_token_idx); // 1, batch*seq, 1, hidden

            // step.4 - 释放专家内存
            experts[i].free();

            string expert_name = std::to_string(layer_idx) + "_" + std::to_string(i);
            auto time_end__ = (mllm_time_us());                    // ms
            double time_end = (time_end__ - start_time) / 1000.0F; // ms
            expert_cal_times[expert_name] = {time_start, time_end};

#ifdef MBP_THREAD
            dones[layer_idx][i] = false; // 重置状态
#endif

            start_idx = end_idx;
        }

        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {expert_cache};
    }

    void load_experts(int expert_idx) {
        experts[expert_idx].load();
    }

private:
    std::vector<SmallThinkerMLP> experts;
    // Layer primary_router;
    Layer sigmoid;
    int num_experts_per_tok{};
    int num_hidden_layers{};
};

class SmallThinkerDecoder final : public Module {
public:
    SmallThinkerDecoder() = default;
    SmallThinkerDecoder(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        SPLIT_NONE, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, false,
                                        config.attn_implementation, names, base_name + names._attn_base_name);
        block_sparse_moe = SmallThinkerMoeBlock(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        num_hidden_layers = config.num_hidden_layers;
        primary_router = Linear(config.hidden_size, config.num_experts, false, base_name + names._ffn_base_name + "primary_router");
        num_experts_per_tok = config.num_experts_per_tok;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto router_input = inputs[0];
        int layer_idx = std::any_cast<int>(args[0]);
        if (router_input.batch() > 1) router_input = router_input.view(1, -1, 1, -1); // 1, batch*seq, 1, hidden
        auto router_logits = primary_router(router_input);                            //  1, batch*seq, 1, num_experts
        auto experts_w_i = Tensor::topk(router_logits, num_experts_per_tok, DIMENSION);
        router_logits = experts_w_i[0];                     //  1, batch*seq, 1, k
        auto expert_indices = experts_w_i[1];               //  1, batch*seq, 1, k
        expert_indices = expert_indices.view(-1, 1, 1, -1); // 1, 1, 1, k* batch*seq
        if (expert_indices.dimension()) {
            auto start_ptr = expert_indices.ptrAt<float>(0, 0, 0, 0);
            auto ptr_len = expert_indices.dimension();
            std::vector<int> unique_experts;
            std::set<float> seen_experts;
            for (int i = 0; i < ptr_len; ++i) {
                float expert_id = start_ptr[i];
                if (seen_experts.find(expert_id) == seen_experts.end()) {
                    seen_experts.insert(expert_id);
                    unique_experts.push_back((int)expert_id);
                }
            }
            std::sort(unique_experts.begin(), unique_experts.end());
            for (int e_i = 0; e_i < unique_experts.size(); ++e_i) {
                auto expert_id = unique_experts[e_i];
                // 向加载队列申请加载 layer_idx的 expert_id专家
#ifdef MBP_THREAD
                LoadRequest req{layer_idx, expert_id};
                {
                    lock_guard<mutex> lk(queue_mutex);
                    load_requests.push(req);
                }
                queue_cv.notify_one(); // 通知加载线程
#endif
                //  std::cout << "layer " << layer_idx << " Request loading expert id: " << expert_id << std::endl;
            }
        }
        auto hidden_states = input_layernorm(router_input);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto residual = hidden_states + inputs[0];
        hidden_states = post_attention_layernorm(residual);
        hidden_states = block_sparse_moe({hidden_states, router_logits, expert_indices}, layer_idx)[0];
        hidden_states = hidden_states + residual;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

    void load_experts(int expert_idx) {
        block_sparse_moe.load_experts(expert_idx);
    }

private:
    MultiHeadAttention self_atten;
    SmallThinkerMoeBlock block_sparse_moe;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    Layer primary_router;
    int num_hidden_layers;
    int num_experts_per_tok{};
};

class SmallThinkerModel final : public Module {
public:
    SmallThinkerModel() = default;
    SmallThinkerModel(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        blocks = List<SmallThinkerDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        int layer_idx = 0;
        for (auto &block : blocks) {
            hidden_states = block({hidden_states}, layer_idx)[0];
            layer_idx++;
        }
        hidden_states = norm(hidden_states);
        return {hidden_states};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }

    void load_experts(int layer_idx, int expert_idx) {
        blocks[layer_idx].load_experts(expert_idx);
    }

private:
    std::vector<SmallThinkerDecoder> blocks;
    Layer norm;
};

class SmallThinkerForCausalLM final : public Module {
public:
    CHAINABLE_MODULE_METHODS(SmallThinkerForCausalLM)
    SmallThinkerForCausalLM(SmallThinkerConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = SmallThinkerModel(config, names, names.blk_name);
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
        } else {
            lm_head_layer = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
        }

        // 初始化 mbp 相关变量
        // mbp_init(config.num_hidden_layers, config.num_experts);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        std::vector<Tensor> outputs;
        clearMBPtimes();
#ifdef MBP_THREAD
        start_time = mllm_time_us();
        mbp_finish.store(false, std::memory_order_relaxed);
        if (inputs[0].dimension() == 1) {
            omp_set_max_active_levels(2); // Enable OpenMP nesting
#pragma omp parallel num_threads(2)
            if (omp_get_thread_num() == 0) { // 根据线程ID决定执行哪个函数
#if defined(__ARM_NEON) && !defined(__APPLE__)
                {
                    struct sched_param param;
                    param.sched_priority = 20; // 范围 1–99，根据设备可酌情调整
                    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
                }
                // ─── 2. 绑定到大核（big cluster）以减少与小核的资源争用 ──────────────
                {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    // 假设大核是 CPU 2–3，按实际设备改为合适的核号
                    CPU_SET(2, &cpuset);
                    CPU_SET(3, &cpuset);
                    // CPU_SET(6, &cpuset); // 假设小核心是CPU 6
                    sched_setaffinity(pthread_self(), sizeof(cpuset), &cpuset);
                    // sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
                }
#endif
                mbp_load();
            } else {
                outputs = do_Forward(inputs, args);
            }
        } else {
#endif
            outputs = do_Forward(inputs, args);
#ifdef MBP_THREAD
        }
#endif
        return outputs;
    }
    void clear_kvcache() override {
        model.clear_kvcache();
    }

    std::vector<Tensor> do_Forward(std::vector<Tensor> inputs, std::vector<std::any> args) {
        auto x = embedding(inputs[0]);
        auto outputs = model({x})[0];
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
        if (tie_embedding_words) {
            outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            outputs = lm_head_layer(outputs);
        }

#ifdef MBP_THREAD
        //  设置 mbp_finish 为 true，结束 mbp_load 线程
        //  1. 设置内存序保证可见性
        mbp_finish.store(true, std::memory_order_release); // 改为 release 内存序
        // 2. 主动唤醒所有等待线程
        {
            std::lock_guard<std::mutex> lk(queue_mutex);
            queue_cv.notify_all(); // 必须加锁后通知
        }
        // 3. 添加二次状态检查（可选）
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
        return {outputs};
    }

    void load_experts(int layer_idx, int expert_idx) {
        model.load_experts(layer_idx, expert_idx);
    }

    void mbp_load() {
        while (!mbp_finish.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lk(queue_mutex);
            queue_cv.wait(lk, [this] {
                return !load_requests.empty() || mbp_finish.load(std::memory_order_acquire);
            });

            if (mbp_finish.load(std::memory_order_acquire)) {
                break;
            }

            while (!load_requests.empty()) {
                auto req = load_requests.front();
                load_requests.pop();
                lk.unlock(); // 释放锁以便其他线程入队
                {            // 执行加载
                    std::unique_lock<std::mutex> expert_lk(*mtxs[req.layer][req.expert]);
                    if (!dones[req.layer][req.expert].load(std::memory_order_acquire)) {
                        double time_start = (mllm_time_us() - start_time) / 1000.0F; // ms

                        load_experts(req.layer, req.expert);
                        dones[req.layer][req.expert].store(true, std::memory_order_release);

                        string expert_name = std::to_string(req.layer) + "_" + std::to_string(req.expert);
                        double time_end = (mllm_time_us() - start_time) / 1000.0F; // ms
                        load_times[expert_name] = {time_start, time_end};
                    }
                }
                cvs[req.layer][req.expert]->notify_all();
                lk.lock(); // 重新获取锁处理下一个请求
            }
        }
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    SmallThinkerModel model;
};

#endif // MODELING_SMOLTHINKER_HPP