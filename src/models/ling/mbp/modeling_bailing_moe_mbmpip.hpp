#pragma once
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "../configuration_bailing_moe.hpp"
#include "settings_bailing_moe_mbp.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>

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
        if (num_shared_experts > 0) {
            shared_experts = BailingMoeMLP(config.hidden_size,
                                           config.moe_intermediate_size * config.num_shared_experts,
                                           names, base_name + "shared_experts.");
        }
        num_hidden_layers = config.num_hidden_layers;
    }
    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        int layer_idx = std::any_cast<int>(args[0]);
        auto hidden_states = inputs[0];
        auto identity = hidden_states;
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto gates_t = gate({hidden_states});                                       //  1, batch*seq, 1, num_experts
        auto scores = gates_t[0];                                                   // 1, batch*seq, 1, num_experts
        auto topk_weight = gates_t[1];                                              // 1, batch*seq,
        auto topk_idx = gates_t[2];                                                 // 1, batch*seq, 1, k
        hidden_states = moe_infer(hidden_states, topk_weight, topk_idx, layer_idx); // 1, batch*seq, 1, hidden
        if (num_shared_experts) {
            hidden_states = hidden_states + shared_experts({identity})[0]; // add shared experts
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {hidden_states};
    }
    Tensor moe_infer(Tensor hidden_states,
                     Tensor &topk_weight,
                     Tensor &topk_idx,
                     int layer_idx) {
        auto idxs = topk_idx.argsort();
        auto tokens_per_expert = topk_idx.bincount();
        auto token_idxs = idxs / num_experts_per_tok;
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states);
        map<int, Tensor> exp_token_idx_list, exp_idx_list;
        std::vector<int> sorted_keys;
        for (int i = 0; i < experts.size(); ++i) {
            if (i >= tokens_per_expert.dimension()) break;
            int this_token_num = tokens_per_expert.dimension() ? tokens_per_expert.d<float>(0, 0, 0, i) : 0;
            if (!this_token_num) continue;
            end_idx = start_idx + this_token_num;
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx});
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});
            topk_weight = topk_weight.view(-1, -1, 1, 1);
            exp_token_idx_list[i] = exp_token_idx;
            sorted_keys.push_back(i);
            exp_idx_list[i] = exp_idx;
            start_idx = end_idx;
        }
        if (!sorted_keys.empty()) {
            int mv_i = 0;
            if (std::find(sorted_keys.begin(), sorted_keys.end(), mv_i) != sorted_keys.end()) {
                sorted_keys.erase(std::remove(sorted_keys.begin(), sorted_keys.end(), mv_i), sorted_keys.end());
                sorted_keys.insert(sorted_keys.begin(), mv_i);
            }
        }

        if (sorted_keys.empty() || Module::doLoad) {
            return expert_cache;
        }

#ifdef MBP_THREAD
        // 步骤 1: 启动流水线 - 预先为第一个专家派发任务
        {
            int first_expert_id = sorted_keys[0];
            // 派发加载任务
            if (!experts[first_expert_id].loaded()) {
                LoadRequest req{layer_idx, first_expert_id};
                lock_guard<mutex> lk(queue_mutex);
                load_requests.push(req);
                queue_cv.notify_one();
            }
            // 派发裁剪任务
            ClipRequest req{
                layer_idx, first_expert_id, hidden_states,
                exp_token_idx_list[first_expert_id], topk_weight, exp_idx_list[first_expert_id]};
            lock_guard<mutex> lk(clip_queue_mutex);
            clip_requests.push(req);
            clip_queue_cv.notify_one();
        }
#endif

        // 步骤 2: 循环处理
        for (int ii = 0; ii < sorted_keys.size(); ii++) {
            int expert_id = sorted_keys[ii];
            string expert_name = std::to_string(layer_idx) + "_" + std::to_string(expert_id);

#ifdef MBP_THREAD
            // A. [预取] 为下一个专家 (ii+1) 派发任务
            bool is_last_expert_in_layer = (ii == sorted_keys.size() - 1);
            if (!is_last_expert_in_layer || (is_last_expert_in_layer && layer_idx < num_hidden_layers - 1)) {
                int q_layer_idx, q_expert_id;
                bool should_dispatch_clip = false;

                if (is_last_expert_in_layer) { // 如果是本层最后一个，预取下一层的 expert 0
                    q_layer_idx = layer_idx + 1;
                    q_expert_id = 0;
                    // 对于下一层的专家，我们无法知道它是否有token，因此不派发裁剪任务
                    should_dispatch_clip = false;
                } else { // 否则，预取本层的下一个专家
                    q_layer_idx = layer_idx;
                    q_expert_id = sorted_keys[ii + 1];
                    // 仅当该专家确实需要处理时，才派发裁剪任务
                    should_dispatch_clip = exp_token_idx_list.count(q_expert_id) > 0;
                }

                // 派发加载任务
                LoadRequest load_req{q_layer_idx, q_expert_id};
                lock_guard<mutex> load_lk(queue_mutex);
                load_requests.push(load_req);
                queue_cv.notify_one();

                // 根据判断条件派发裁剪任务
                if (should_dispatch_clip) {
                    ClipRequest clip_req{
                        q_layer_idx, q_expert_id, hidden_states,
                        exp_token_idx_list[q_expert_id], topk_weight, exp_idx_list[q_expert_id]};
                    lock_guard<mutex> clip_lk(clip_queue_mutex);
                    clip_requests.push(clip_req);
                    clip_queue_cv.notify_one();
                }
            }

            // B. [等待] 等待当前专家 (ii) 的任务完成
            // 等待加载
            double time_start_w = (mllm_time_us() - start_time) / 1000.0F;
            if (!experts[expert_id].loaded()) {
                unique_lock<mutex> lock(*mtxs[layer_idx][expert_id]);
                cvs[layer_idx][expert_id]->wait(lock, [&] { return dones[layer_idx][expert_id].load(memory_order_acquire); });
            }
            double time_end_w = (mllm_time_us() - start_time) / 1000.0F;
            expert_wait_times[expert_name] = {time_start_w, time_end_w};

            // 等待裁剪并获取结果
            Tensor expert_tokens, topk_weight_clip;
            {
                unique_lock<mutex> lock(*clip_mtxs[layer_idx][expert_id]);
                clip_cvs[layer_idx][expert_id]->wait(lock, [&] { return clip_dones[layer_idx][expert_id].load(memory_order_acquire); });

                std::lock_guard<std::mutex> result_lk(clip_results_mutex);
                auto &clipped_pair = clipped_data.at(expert_name);
                expert_tokens = clipped_pair.first;
                topk_weight_clip = clipped_pair.second;
            }
#else
            // 非多线程模式，直接加载和裁剪
            if (!experts[expert_id].loaded()) experts[expert_id].load();
            auto expert_tokens = hidden_states.clip(exp_token_idx_list[expert_id], SEQUENCE);
            auto topk_weight_clip = topk_weight.clip(exp_idx_list[expert_id], SEQUENCE);
#endif

            // C. [计算] 使用准备好的数据进行计算
            double time_start_cal = (mllm_time_us() - start_time) / 1000.0F;
            auto expert_out = experts[expert_id]({expert_tokens})[0];
            expert_out = expert_out * topk_weight_clip;
            expert_cache.scatter_add(expert_out, exp_token_idx_list[expert_id]);
            double time_end_cal = (mllm_time_us() - start_time) / 1000.0F;
            expert_cal_times[expert_name] = {time_start_cal, time_end_cal};

            // D. [清理] 清理当前专家的资源
            experts[expert_id].free();
#ifdef MBP_THREAD
            {
                std::lock_guard<std::mutex> result_lk(clip_results_mutex);
                clipped_data.erase(expert_name);
            }
            // std::cout << clipped_data.size() << std::endl;
            clip_dones[layer_idx][expert_id] = false;
            dones[layer_idx][expert_id] = false;
#endif
        }
        return expert_cache;
    }

    void load_experts(int expert_idx) {
        int result;
        experts[expert_idx].load();
    }

private:
    BailingMoeMLP shared_experts;
    std::vector<BailingMoeMLP> experts;
    BailingMoeGate gate;
    int num_shared_experts{};
    int num_experts_per_tok{};
    int num_hidden_layers{};
};

class BailingMoeDecoder final : public Module {
public:
    BailingMoeDecoder() = default;
    BailingMoeDecoder(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        SPLIT_HD, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                                        config.cache_limit, config.use_cache, config.use_qkv_bias, config.use_bias,
                                        config.attn_implementation, names, base_name + names._attn_base_name);
        moe = BailingMoeSparseMoeBlock(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = input_layernorm(inputs[0]);
        int layer_idx = std::any_cast<int>(args[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states + inputs[0];
        hidden_states = post_attention_layernorm(tmp);
        hidden_states = moe({hidden_states}, layer_idx)[0];
        hidden_states = hidden_states + tmp;
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
        int layer_idx = 0;
        for (auto &block : blocks) {
            hidden_states = block({hidden_states}, layer_idx)[0];
            layer_idx++;
        }
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
    std::vector<BailingMoeDecoder> blocks;
    Layer norm;
};

class BailingMoeForCausalLM final : public Module {
public:
    BailingMoeForCausalLM(BailingMoeConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = BailingMoeModel(config, names, names.blk_name);
        lm_head = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        std::vector<Tensor> outputs;
        clearMBPtimes();
#ifdef MBP_THREAD
        start_time = mllm_time_us();
        mbp_finish.store(false, std::memory_order_relaxed);
        if (inputs[0].dimension() == 1) {
            omp_set_max_active_levels(2); // Enable OpenMP nesting
#pragma omp parallel num_threads(3)
            if (omp_get_thread_num() == 0) { // 根据线程ID决定执行哪个函数
#if defined(__ARM_NEON) && !defined(__APPLE__)
                {
                    struct sched_param param;
                    param.sched_priority = 21; // 范围 1–99，根据设备可酌情调整
                    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
                }
                // ─── 2. 绑定到大核（big cluster）以减少与小核的资源争用 ──────────────
                {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(2, &cpuset);
                    sched_setaffinity(pthread_self(), sizeof(cpuset), &cpuset);
                }
#endif
                mbp_load();
            } else if (omp_get_thread_num() == 1) { // 线程1: 裁剪 (新增)

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
                    CPU_SET(3, &cpuset);
                    sched_setaffinity(pthread_self(), sizeof(cpuset), &cpuset);
                }
#endif
                mbp_clip();
            } else {
                // #if defined(__ARM_NEON) && !defined(__APPLE__)
                //                 {
                //                     struct sched_param param;
                //                     param.sched_priority = 22; // 范围 1–99，根据设备可酌情调整
                //                     pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
                //                 }
                //                 // ─── 2. 绑定到大核（big cluster）以减少与小核的资源争用 ──────────────
                //                 {
                //                     cpu_set_t cpuset;
                //                     CPU_ZERO(&cpuset);
                //                     CPU_SET(7, &cpuset);
                //                     sched_setaffinity(pthread_self(), sizeof(cpuset), &cpuset);
                //                 }
                // #endif
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
        outputs = lm_head(outputs);

#ifdef MBP_THREAD
        //  设置 mbp_finish 为 true，结束 mbp_load 线程
        //  1. 设置内存序保证可见性
        mbp_finish.store(true, std::memory_order_release); // 改为 release 内存序
        // 2. 主动唤醒所有等待线程
        {
            std::lock_guard<std::mutex> lk(queue_mutex);
            queue_cv.notify_all(); // 必须加锁后通知
        }
        {
            std::lock_guard<std::mutex> lk(clip_queue_mutex);
            clip_queue_cv.notify_all(); // [新增] 唤醒 clip 线程
        }
        // 3. 添加二次状态检查
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // std::cout << "do_Forward finish  " << load_requests.size() << std::endl;
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
    void mbp_clip() {
        while (!mbp_finish.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lk(clip_queue_mutex);
            clip_queue_cv.wait(lk, [this] {
                return !clip_requests.empty() || mbp_finish.load(std::memory_order_acquire);
            });

            if (mbp_finish.load(std::memory_order_acquire)) {
                break;
            }

            while (!clip_requests.empty()) {
                auto req = clip_requests.front();
                clip_requests.pop();
                lk.unlock();

                string expert_name = std::to_string(req.layer) + "_" + std::to_string(req.expert);

                // --- 执行裁剪 ---
                double time_start_ = (mllm_time_us() - start_time) / 1000.0F;
                auto expert_tokens = req.hidden_states.clip(req.exp_token_idx, SEQUENCE);
                auto topk_weight_clip = req.topk_weight.clip(req.exp_idx, SEQUENCE);
                double time_end_ = (mllm_time_us() - start_time) / 1000.0F;
                expert_clip_times[expert_name] = {time_start_, time_end_};

                // --- 存储结果 ---
                {
                    std::lock_guard<std::mutex> result_lk(clip_results_mutex);
                    clipped_data[expert_name] = {expert_tokens, topk_weight_clip};
                }

                // --- 发送完成信号 ---
                {
                    std::unique_lock<std::mutex> done_lk(*clip_mtxs[req.layer][req.expert]);
                    clip_dones[req.layer][req.expert].store(true, std::memory_order_release);
                }
                clip_cvs[req.layer][req.expert]->notify_all();

                lk.lock();
            }
        }
        // std::cout << "mbp_clip finish" << std::endl;
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Layer lm_head;
    BailingMoeModel model;
};
