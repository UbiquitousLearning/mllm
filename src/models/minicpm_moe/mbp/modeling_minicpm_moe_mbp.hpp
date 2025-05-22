#ifndef MODELING_MINICPMMOE_MBP_HPP
#define MODELING_MINICPMMOE_MBP_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../configuration_minicpm_moe.hpp"
#include "settings_minicpm_moe_mbp.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <atomic>
#include <cmath>
// #include <thread>
#include <unistd.h>
#include <sys/syscall.h>
// #include <iostream>
#include <vector>
#if defined(__ARM_NEON)
#include <sys/syscall.h>
#endif

using namespace mllm;

#define MBP_THREAD
// #define MBP_THREAD_PP

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

class MiniCPMMoE final : public Module {
public:
    MiniCPMMoE() = default;
    MiniCPMMoE(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        num_hidden_layers = config.num_hidden_layers;
        experts = List<MiniCPMMLP>(config.num_experts, config.hidden_size, config.intermediate_size, names, base_name + "experts.");
        gate = Linear(config.hidden_size, config.num_experts, false, base_name + "gate");
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        num_experts_per_tok = config.num_experts_per_tok;
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        int layer_idx = std::any_cast<int>(args[0]);
        auto hidden_states = inputs[0];
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto scores = gate(hidden_states); //  1, batch*seq, 1, num_experts
        scores = softmax(scores);
        auto experts_w_i = Tensor::topk(scores, num_experts_per_tok, DIMENSION);
        auto expert_weights = experts_w_i[0];                            //  1, batch*seq, 1, k
        auto expert_indices = experts_w_i[1];                            //  1, batch*seq, 1, k
        expert_indices = expert_indices.view(-1, 1, 1, -1);              // 1, 1, 1, k* batch*seq
        expert_weights = expert_weights / expert_weights.sum(DIMENSION); //  1, batch*seq, 1, k
        expert_weights = expert_weights.view(-1, -1, 1, 1);              // 1, k* batch*seq, 1, 1
        // moe_infer
        auto idxs = expert_indices.argsort();               // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = expert_indices.bincount(); // (1, 1, 1, 0) 1, 1, 1, k
        auto token_idxs = idxs / num_experts_per_tok;       // 1, 1, 1, k* batch*seq
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states); // 1, batch*seq, 1, hidden
        map<int, Tensor> exp_token_idx_list, exp_idx_list;
        std::vector<int> sorted_keys; // 根据 exp_token_idx_list[i].dimension() 对键值排序
        for (int i = 0; i < experts.size(); ++i) {
            if (i >= tokens_per_expert.dimension()) break; // 全部专家计算完
            int this_token_num = tokens_per_expert.dimension() ? tokens_per_expert.d<float>(0, 0, 0, i) : 0;
            if (!this_token_num) continue;
            end_idx = start_idx + this_token_num;
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx}); //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s
            exp_token_idx_list[i] = exp_token_idx;
            sorted_keys.push_back(i);
            exp_idx_list[i] = exp_idx;
            start_idx = end_idx;
        }
        std::sort(sorted_keys.begin(), sorted_keys.end(), [&](int a, int b) {
            return exp_token_idx_list[a].dimension() > exp_token_idx_list[b].dimension();
        });
        if (!sorted_keys.empty()) {
            int mv_i = 0;
            if (std::find(sorted_keys.begin(), sorted_keys.end(), mv_i) != sorted_keys.end()) {
                sorted_keys.erase(std::remove(sorted_keys.begin(), sorted_keys.end(), mv_i), sorted_keys.end());
                sorted_keys.insert(sorted_keys.begin(), mv_i);
            }

            if (!experts[sorted_keys[0]].loaded()) {
                double time_start = (mllm_time_us() - start_time) / 1000.0F; // ms

                experts[sorted_keys[0]].load();

                string expert_name = std::to_string(layer_idx) + "_" + std::to_string(sorted_keys[0]);
                double time_end = (mllm_time_us() - start_time) / 1000.0F; // ms
                load_times[expert_name] = {time_start, time_end};
                // std::cout << "load: " << layer_idx << " " << sorted_keys[0] << std::endl;
            }

            // std::cout << layer_idx << "_sorted_keys [";
            for (auto s : sorted_keys) {
                // std::cout << s << " ";
            }
            // std::cout << "]" << std::endl;
        }
        for (int ii = 0; ii < sorted_keys.size(); ii++) {
            int expert_id = sorted_keys[ii];
            if (exp_token_idx_list.find(expert_id) == exp_token_idx_list.end()) continue; // 退出
            if (Module::doLoad) continue;                                                 // 退出

            // step.0
            if ((ii < sorted_keys.size() - 1 && exp_token_idx_list[sorted_keys[ii + 1]].dimension() > 0)
                || (ii == sorted_keys.size() - 1 && layer_idx < num_hidden_layers - 1)) {
#ifdef MBP_THREAD
                int q_layer_idx, q_expert_id;
                if (ii == sorted_keys.size() - 1 && layer_idx < num_hidden_layers - 1) {
                    q_layer_idx = layer_idx + 1;
                    q_expert_id = 0;
                } else {
                    q_layer_idx = layer_idx;
                    q_expert_id = sorted_keys[ii + 1];
                }
                LoadRequest req{q_layer_idx, q_expert_id};
                {
                    lock_guard<mutex> lk(queue_mutex);
                    load_requests.push(req);
                    // std::cout << "load_requests.push: " << q_layer_idx << " " << q_expert_id << std::endl;
                }
                queue_cv.notify_one(); // 通知加载线程
#else
                if (ii < sorted_keys.size() - 1 && exp_token_idx_list[sorted_keys[ii + 1]].dimension() > 0) {
                    auto time_start___ = (mllm_time_us());                      // ms
                    double time_start = (time_start___ - start_time) / 1000.0F; // ms

                    experts[sorted_keys[ii + 1]].load();

                    dones[layer_idx][sorted_keys[ii + 1]].store(true, std::memory_order_release);
                    string expert_name = std::to_string(layer_idx) + "_" + std::to_string(sorted_keys[ii + 1]);

                    auto time_end__ = (mllm_time_us());                    // ms
                    double time_end = (time_end__ - start_time) / 1000.0F; // ms
                    load_times[expert_name] = {time_start, time_end};
                    double tt_t = (time_end__ - time_start___) / 1000.0F;
                    // std::cout << "load: " << layer_idx << " " << sorted_keys[ii + 1] << " " << tt_t << std::endl;
                }
#endif
            }
#if defined(MBP_THREAD) && defined(MBP_THREAD_PP)
        }
        for (int ii = 0; ii < sorted_keys.size(); ii++) {
            int expert_id = sorted_keys[ii];
            if (exp_token_idx_list.find(expert_id) == exp_token_idx_list.end()) continue; // 退出
#endif

            // step.1
            double time_start_ = (mllm_time_us() - start_time) / 1000.0F; // ms

            auto exp_token_idx = exp_token_idx_list[expert_id];                //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = exp_idx_list[expert_id];                            //(1, 1, 1, 0) 1, 1, 1, e-s
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE);  //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto expert_weights_clip = expert_weights.clip(exp_idx, SEQUENCE); //(1, 0, 1, 1) 1, e-s, 1, 1

            string expert_name_ = std::to_string(layer_idx) + "_" + std::to_string(expert_id);
            double time_end_ = (mllm_time_us() - start_time) / 1000.0F; // ms
            expert_clip_times[expert_name_] = {time_start_, time_end_};

#ifdef MBP_THREAD
            // std::cout << "wait: " << layer_idx << " " << expert_id << std::endl;
            if (layer_idx + ii > 0 && !experts[expert_id].loaded()) {
                // std::cout << "wait-: " << layer_idx << " " << expert_id << std::endl;
                unique_lock<mutex> lock(*mtxs[layer_idx][expert_id]); // 局部锁
                cvs[layer_idx][expert_id]->wait(lock, [&] {
                    return dones[layer_idx][expert_id].load(memory_order_acquire);
                });
                assert(dones[layer_idx][expert_id]);
            }
            // std::cout << "waited: " << layer_idx << " " << expert_id << std::endl;
#endif
            auto time_start__ = (mllm_time_us());                      // ms
            double time_start = (time_start__ - start_time) / 1000.0F; // ms

            // step.2
            auto expert_out = experts[expert_id]({expert_tokens})[0]; //(1, 0, 1, hidden) 1, e-s, 1,
            expert_out = expert_out * expert_weights_clip;            //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_reduce(expert_out, exp_token_idx);   // 1, batch*seq, 1, hidden
            experts[expert_id].free();
            // std::cout << "free: " << layer_idx << " " << expert_id << std::endl;

            string expert_name = std::to_string(layer_idx) + "_" + std::to_string(expert_id);
            auto time_end__ = (mllm_time_us());                    // ms
            double time_end = (time_end__ - start_time) / 1000.0F; // ms
            expert_cal_times[expert_name] = {time_start, time_end};
            // std::cout << "calc: " << layer_idx << " " << expert_id << " " << (time_end__ - time_start__) / (1000.0F * expert_tokens.sequence()) << std::endl;
#ifdef MBP_THREAD
            // std::cout << "dones: " << layer_idx << " " << expert_id << std::endl;
            dones[layer_idx][expert_id] = false; // 重置状态
#endif
            /*
            // step.3
            if ((ii < sorted_keys.size() - 1 && exp_token_idx_list[sorted_keys[ii + 1]].dimension() > 0)
                || (ii == sorted_keys.size() - 1 && layer_idx < num_hidden_layers - 1)) {
#ifdef MBP_THREAD
                int q_layer_idx, q_expert_id;
                if (ii == sorted_keys.size() - 1 && layer_idx < num_hidden_layers - 1) {
                    q_layer_idx = layer_idx + 1;
                    q_expert_id = 0;
                } else {
                    q_layer_idx = layer_idx;
                    q_expert_id = sorted_keys[ii + 1];
                }
                LoadRequest req{q_layer_idx, q_expert_id};
                {
                    lock_guard<mutex> lk(queue_mutex);
                    load_requests.push(req);
                    // std::cout << "load_requests.push: " << q_layer_idx << " " << q_expert_id << std::endl;
                }
                queue_cv.notify_one(); // 通知加载线程
#else
                static double tt_t = 0;
                if (ii < sorted_keys.size() - 1 && exp_token_idx_list[sorted_keys[ii + 1]].dimension() > 0) {
                    // auto time_start___ = (mllm_time_us()); // ms

                    // std::cout << experts[sorted_keys[ii + 1]].loaded() << std::endl;
                    experts[sorted_keys[ii + 1]].load();

                    // auto time_end__ = (mllm_time_us()); // ms
                    // tt_t += (time_end__ - time_start___) / 1000.0F;
                    // std::cout << "load tt_t: " << tt_t << std::endl;
                    // std::cout << "load: " << layer_idx << " " << sorted_keys[ii + 1] << " " << time_end - time_start << std::endl;
                }
#endif
            }
            */
        }
        // std::cout << "finish: " << layer_idx << std::endl;

        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);
        }
        return {expert_cache};
    }

    void load_experts(int expert_idx) {
        int result;
        experts[expert_idx].load();
    }

private:
    std::vector<MiniCPMMLP> experts;
    Layer gate;
    Softmax softmax;

    int num_experts_per_tok{};
    int num_hidden_layers{};
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
        int layer_idx = std::any_cast<int>(args[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + inputs[0];
        hidden_states = post_attention_layernorm(tmp);
        hidden_states = moe({hidden_states}, layer_idx)[0];
        hidden_states = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + tmp;
        return {hidden_states};
    }

    void load_experts(int expert_idx) {
        moe.load_experts(expert_idx);
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
        int layer_idx = 0;
        for (auto &block : blocks) {
            hidden_states = block({hidden_states}, layer_idx)[0];
            layer_idx++;
        }
        // std::cout << "b_finish: " << layer_idx << " |" << blocks.size() << std::endl;
        hidden_states = norm(hidden_states);
        return {hidden_states};
    }

    void load_experts(int layer_idx, int expert_idx) {
        blocks[layer_idx].load_experts(expert_idx);
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
        num_hidden_layers = config.num_hidden_layers;
        num_experts = config.num_experts;
        // KVCache_TYPE = 32;
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
#ifdef MBP_THREAD
        start_time = mllm_time_us();
        mbp_finish.store(false, std::memory_order_relaxed);
        if (inputs[0].dimension() == 1) {
            // OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
            // omp_set_nested(1);            // 等价于设置环境变量 OMP_NESTED=TRUE
            omp_set_max_active_levels(2); // Enable OpenMP nesting
#pragma omp parallel num_threads(2)
            if (omp_get_thread_num() == 0) { // 根据线程ID决定执行哪个函数
#if defined(__ARM_NEON)
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
    std::vector<Tensor> do_Forward(std::vector<Tensor> inputs, std::vector<std::any> args) {
        auto x = embedding(inputs[0]) * scale_emb;
        auto outputs = model({x})[0];
        outputs = outputs / (hidden_size / dim_model_base);
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));

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
        // std::cout << "do_Forward finish  " << load_requests.size() << std::endl;
#endif
        return {outputs};
    }
    void load_experts(int layer_idx, int expert_idx) {
        model.load_experts(layer_idx, expert_idx);
    }
    void clear_kvcache() override {
        model.clear_kvcache();
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

                        // std::cout << "load_requests.load_: " << req.layer << " " << req.expert << std::endl;
                        load_experts(req.layer, req.expert);
                        // std::cout << "load_requests.load_d: " << req.layer << " " << req.expert << std::endl;
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
        // std::cout << "mbp_load finish" << std::endl;
    }

private:
    int hidden_size;
    float dim_model_base;
    bool tie_embedding_words;
    float scale_emb;
    Layer embedding;
    Parameter lm_head;
    MiniCPMModel model;
    int num_hidden_layers;
    int num_experts;
};

#endif // MODELING_MINICPMMOE_MBP_HPP