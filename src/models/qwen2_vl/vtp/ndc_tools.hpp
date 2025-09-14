//
// Created by Rongjie Yi on 25-5-29.
//
#ifndef NDC_TOOLS_HPP
#define NDC_TOOLS_HPP

#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace mllm;

class DelayComputeKVCache {
public:
    vector<vector<int>> kv_true_token_appds;
    vector<Tensor> hidden_states_cache;
    vector<Tensor> hidden_states_filled;
    DelayComputeKVCache() {
    }
    void init_cache_list(int layers) {
        hidden_states_cache.resize(layers);
        hidden_states_filled.resize(layers);
        kv_true_token_appds.resize(layers);
    }
    void update_hidden_states(Tensor hidden_states, int layer_idx, int original_hs_length, vector<int> pos, bool is_prefill) {
        auto b = hidden_states.batch();
        auto d = hidden_states.dimension();
        if (hidden_states_cache[layer_idx].name().empty()) { //=如果hidden_states_cache[layer_idx]为空Tensor
            hidden_states_cache[layer_idx] = Tensor(b, 1, original_hs_length, d, MLLM_CPU, true);
            hidden_states_cache[layer_idx].setName("hidden_states_cache_" + std::to_string(layer_idx));
            hidden_states_filled[layer_idx] = Tensor(b, 1, original_hs_length, 1, MLLM_CPU, true);
            hidden_states_filled[layer_idx].setName("hidden_states_fille_" + std::to_string(layer_idx));
        }
        for (int bb = 0; bb < b; ++bb) {
            for (int i = 0; i < pos.size(); ++i) {
                auto p = pos[i];
                memcpy(hidden_states_cache[layer_idx].ptrAt<float>(bb, 0, p, 0),
                       hidden_states.ptrAt<float>(bb, 0, i, 0),
                       sizeof(float) * d);
                hidden_states_filled[layer_idx].setDataAt<float>(bb, 0, p, 0, 1.0f);
            }
        }
    }
    Tensor get_hidden_states(int layer_idx, vector<int> pos) {
        return hidden_states_cache[layer_idx].clip(pos, SEQUENCE);
    }
    Tensor reset_hidden_states(Tensor hidden_states, int layer_idx, vector<int> pos) {
        assert(hidden_states.batch() == 1);
        vector<float> hidden_states_last;
        hidden_states_last.resize(hidden_states.dimension());
        memcpy(hidden_states_last.data(),
               hidden_states.ptrAt<float>(0, 0, hidden_states.sequence() - 1, 0),
               sizeof(float) * hidden_states.dimension());
        hidden_states.reshape(1, 1, pos.size() + 1, hidden_states.dimension());
        hidden_states.alloc();
        for (int i = 0; i < pos.size(); ++i) {
            int p = pos[i];
            memcpy(hidden_states.ptrAt<float>(0, 0, i, 0),
                   hidden_states_cache[layer_idx].ptrAt<float>(0, 0, p, 0),
                   sizeof(float) * hidden_states.dimension());
        }
        memcpy(hidden_states.ptrAt<float>(0, 0, hidden_states.sequence() - 1, 0),
               hidden_states_last.data(),
               sizeof(float) * hidden_states.dimension());
        return hidden_states;
    }
    vector<int> kv_not_filled_pos(int layer_idx, int original_kv_length) {
        auto filled_token = kv_true_token_appds[layer_idx];
        vector<int> not_filled_pos;
        for (int i = 0; i < original_kv_length; ++i) {
            if (std::find(filled_token.begin(), filled_token.end(), i) == filled_token.end()) {
                not_filled_pos.push_back(i);
            }
        }
        return not_filled_pos;
    }
    template <typename T>
    static void reorder_cache(Tensor &k_cache, Tensor &v_cache,
                              const vector<size_t> &indices,
                              int pos_first, int cache_sequence) {
        const int num_heads = v_cache.head();
        const int k_per_head = k_cache.dimension();
        const int v_per_head = v_cache.dimension();
        const int k_size = num_heads * k_per_head;
        const int v_size = v_per_head;
        // 1. 分配临时内存
        if (cache_sequence <= pos_first) {
            pos_first = 0;
        }
        vector<vector<T>> k_cache_data(cache_sequence - pos_first);
        vector<vector<vector<T>>> v_cache_data(num_heads);
        for (int i = pos_first; i < cache_sequence; i++) {
            k_cache_data[i - pos_first].resize(k_size);
        }
        for (int h = 0; h < num_heads; ++h) {
            v_cache_data[h].resize(cache_sequence - pos_first);
            for (int i = pos_first; i < cache_sequence; i++) {
                v_cache_data[h][i - pos_first].resize(v_size);
            }
        }
        // 2. 拷贝数据到临时内存
        for (int i = pos_first; i < cache_sequence; i++) {
            // K_cache拷贝（全部heads）
            memcpy(k_cache_data[i - pos_first].data(),
                   k_cache.ptrAt<T>(0, 0, i, 0),
                   sizeof(T) * k_size);
            // V_cache拷贝（每个head分开）
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int h = 0; h < num_heads; ++h) {
                memcpy(v_cache_data[h][i - pos_first].data(),
                       v_cache.ptrAt<T>(0, h, i, 0),
                       sizeof(T) * v_size);
            }
        }
        // 3. 根据索引重新排序
        for (size_t idx : indices) {
            if (idx >= (size_t)pos_first && idx < (size_t)cache_sequence) {
                const int temp_idx = idx - pos_first;
                // 写回K_cache
                memcpy(k_cache.ptrAt<T>(0, 0, idx, 0),
                       k_cache_data[temp_idx].data(),
                       sizeof(T) * k_size);
                // 写回V_cache
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                for (int h = 0; h < num_heads; ++h) {
                    memcpy(v_cache.ptrAt<T>(0, h, idx, 0),
                           v_cache_data[h][temp_idx].data(),
                           sizeof(T) * v_size);
                }
            }
        }
    }

    void update_kv_cache(Tensor &k_cache, Tensor &v_cache, Tensor &k_state, Tensor &v_state, int cache_sequence, int layer_idx,
                         bool is_prefill, string update_mode, vector<int> pos = {}, int original_kv_length = -1) {
        if (update_mode == "insert") {
            assert(k_cache.masterTensor() == k_state.masterTensor());
            assert(v_cache.masterTensor() == v_state.masterTensor());
            // pos代表现在的token列表：{0，1，2，3，5，6，8，9}， 8个token及其列表
            if (is_prefill) {
                kv_true_token_appds[layer_idx] = pos; // 记录当前token的列表
                assert(kv_true_token_appds[layer_idx].size() == cache_sequence);
            } else {
                auto new_token_pos = kv_true_token_appds[layer_idx][kv_true_token_appds[layer_idx].size() - 1] + 1;
                assert(kv_true_token_appds[layer_idx].size() + 1 + pos.size() == cache_sequence);
                kv_true_token_appds[layer_idx].insert(kv_true_token_appds[layer_idx].end(), pos.begin(), pos.end());
                kv_true_token_appds[layer_idx].push_back(new_token_pos); // 添加新的token位置
                auto &cur_pos = kv_true_token_appds[layer_idx];
                // for k_cache;
                if (pos.size() > 1) {
                    auto pos_first = pos[0];
                    assert(v_cache.ctype() == BHDS);
                    // 创建并初始化索引数组
                    std::vector<size_t> indices(cur_pos.size());
                    std::iota(indices.begin(), indices.end(), 0);
                    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                        return cur_pos[a] < cur_pos[b];
                    });
                    assert(k_cache.batch() == 1);
                    if (k_cache.dtype() == MLLM_TYPE_F16) {
                        reorder_cache<mllm_fp16_t>(k_cache, v_cache, indices, pos_first, cache_sequence);
                    } else {
                        reorder_cache<float>(k_cache, v_cache, indices, pos_first, cache_sequence);
                    }
                    std::sort(kv_true_token_appds[layer_idx].begin(), kv_true_token_appds[layer_idx].end());
                }
            }
        }
    }
};

class NdcContext {
    int first_img_token_pos = 0;
    int last_img_token_pos = 0;
    int last_img_token_pos_l = 0;
    DelayComputeKVCache kvcache_ctx;
    int cur_step = -1;
    vector<vector<int>> chosen_pos_in_each;
    vector<vector<int>> chosen_pos_to_delay_compute;
    Tensor global_position_ids;
    int num_hidden_layers = 0;
    int num_head = 0;
    int original_kv_length = 0;
    int chunk_size = 4;

public:
    // map<int, float> pruning_place_cfg = {{3, 0.2}, {9, 0.2}, {12, 0.6}, {15, 0.6}, {18, 0.8}, {26, 0.8}};
    // map<int, float> pruning_place_cfg = {{3, 0.2}, {6, 0.8}, {12, 0.8}, {15, 0.8}, {18, 0.8}, {26, 0.8}};
    // map<int, float> pruning_place_cfg = {{3, 0.8}, {6, 0.8}, {12, 0.8}, {15, 0.8}, {18, 0.8}, {26, 0.8}};
    map<int, float> pruning_place_cfg = {{6, 0.8}, {12, 0.8}, {18, 0.8}};
    // map<int, float> pruning_place_cfg = {{3, 0.2}, {9, 0.2}};

public:
    map<int, Tensor> causal_masks; // layer_idx -> causal_mask
    bool prefill_stage = true;

    /**
     * @brief Resets the context to its initial state.
     * This function should be called to clear the state between different generation requests.
     */
    void reset() {
        first_img_token_pos = 0;
        last_img_token_pos = 0;
        last_img_token_pos_l = 0;
        kvcache_ctx = DelayComputeKVCache(); // Re-initialize the KV cache context
        cur_step = -1;
        chosen_pos_in_each.clear();
        chosen_pos_to_delay_compute.clear();
        global_position_ids = Tensor(); // Reset to an empty tensor
        num_hidden_layers = 0;
        num_head = 0;
        original_kv_length = 0;
        causal_masks.clear();
        prefill_stage = true;
    }

    void init(Tensor input_ids, int num_layers, int num_attention_heads) {
        if (Module::llm_model_ptr->doLoad) { return; }
        // reset();
        num_hidden_layers = num_layers;
        num_head = num_attention_heads;
        if (kvcache_ctx.hidden_states_cache.empty()) {
            chosen_pos_in_each.resize(num_hidden_layers, {});
            kvcache_ctx.init_cache_list(num_hidden_layers); // Initialize with 1 layer, can be adjusted as needed
        }

        if (input_ids.sequence() <= 1) {
            prefill_stage = false;
        } else {
            prefill_stage = true;
        }
    }
    bool is_prefill() {
        return prefill_stage && cur_step == 0;
    }
    void set_vision_token(Tensor where_idx, Tensor hidden_states, Tensor image_embeds) {
        if (Module::llm_model_ptr->doLoad) { return; }
        first_img_token_pos = int(where_idx.dataAt<float>(0, 0, 0, 0));
        last_img_token_pos = int(where_idx.dataAt<float>(0, 0, 0, where_idx.dimension() - 1)) + 1;
    }

    void ndc_prepare(Tensor hidden_states, Tensor position_ids, int past_kv_seq_len) {
        if (Module::llm_model_ptr->doLoad) { return; }
        cur_step += 1;
        if (cur_step == 0) {
            global_position_ids = position_ids;
            original_kv_length = hidden_states.sequence();
        }
        chosen_pos_in_each.resize(num_hidden_layers, {});
        int new_seq_len = hidden_states.sequence() + past_kv_seq_len;
        chosen_pos_in_each[0].clear();
        for (int i = 0; i < new_seq_len; ++i) {
            chosen_pos_in_each[0].push_back(i);
        }
        if (!is_prefill()) {
            chosen_pos_to_delay_compute.resize(num_hidden_layers, {});
        }
    }
    void get_kvcache(Tensor &k_cache, Tensor &v_cache, Tensor &k_state, Tensor &v_state, int layer_idx, int cache_sequence) {
        if (Module::llm_model_ptr->doLoad) { return; }
        if (is_prefill()) {
            auto chosen_pos = chosen_pos_in_each[layer_idx];
            kvcache_ctx.update_kv_cache(k_cache, v_cache, k_state, v_state, cache_sequence, layer_idx,
                                        is_prefill(), "insert",
                                        chosen_pos, original_kv_length);
        } else {
            auto chosen_pos = chosen_pos_in_each[layer_idx];
            kvcache_ctx.update_kv_cache(k_cache, v_cache, k_state, v_state, cache_sequence, layer_idx,
                                        is_prefill(), "insert",
                                        chosen_pos_to_delay_compute[layer_idx], original_kv_length);
        }
    }

    void topk_partial_sort(const vector<float> &scores, int k,
                           vector<float> &topk_values, vector<int> &topk_indices) {
        if (k <= 0 || scores.empty()) {
            topk_values.clear();
            topk_indices.clear();
            return;
        }
        k = std::min(k, static_cast<int>(scores.size()));
        // 创建索引向量
        vector<int> indices(scores.size());
        for (int i = 0; i < scores.size(); i++) {
            indices[i] = i;
        }
        // 部分排序 - 将前k个最大的元素移动到前部
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&scores](int a, int b) {
                              return scores[a] > scores[b]; // 降序排序
                          });
        // 提取结果
        topk_values.resize(k);
        topk_indices.resize(k);
        for (int i = 0; i < k; i++) {
            topk_indices[i] = indices[i];
            topk_values[i] = scores[indices[i]];
        }
    }

    vector<int> select_high_score_visual_token_prefill(Tensor attn, int layer_idx, int chunk_size = 4) {
        auto cur_chosen_pos = chosen_pos_in_each[layer_idx];
        // attention_score_analyze_prefill start
        attn = attn.mean(HEAD); // 1,t,1,t
        int visual_start_in_selected = -1;
        int visual_end_in_selected = -1;
        for (int i = 0; i < cur_chosen_pos.size(); ++i) {
            auto pos = cur_chosen_pos[i];
            if (pos == first_img_token_pos - 1) { // <visual_start
                visual_start_in_selected = i;
            }
            if (pos == last_img_token_pos) { // <visual_end
                visual_end_in_selected = i;
            }
            if (visual_start_in_selected > 0 && visual_end_in_selected > 0) {
                break;
            }
        }
        int attn_seq_start = visual_end_in_selected + 1;   // +1 for the end image token
        int attn_seq_end = attn.sequence();                // exclusive
        int attn_dim_start = visual_start_in_selected + 1; // +1 for the first image token
        int attn_dim_end = visual_end_in_selected;         // exclusive
        vector<float> attn_score;                          // 1,1,1,visual_end_in_selected - visual_start_in_selected + 1
        for (int j = attn_dim_start; j < attn_dim_end; ++j) {
            float data = 0.0f;
            for (int i = attn_seq_start; i < attn_seq_end; ++i) {
                data += attn.dataAt<float>(0, 0, i, j);
            }
            // data /= (attn_seq_end - attn_seq_start);
            attn_score.push_back(data);
        }
        auto v_s = visual_start_in_selected;
        auto v_e = visual_end_in_selected;
        // attention_score_analyze_prefill end
        auto pruning_rate = pruning_place_cfg[layer_idx];
        auto cur_visual_token_length = attn_score.size();
        auto keep_ratio = 1 - pruning_rate;
        int k_initial = static_cast<int>(std::ceil(cur_visual_token_length * keep_ratio));
        int k_final = (k_initial / chunk_size) * chunk_size;
        k_final = std::min(k_final, static_cast<int>(cur_visual_token_length)); // 确保不超过当前实际长度
        vector<float> topk_vals;
        vector<int> topk_indices;
        topk_partial_sort(attn_score, k_final, topk_vals, topk_indices); // torch.topk(attn_score, k_final)
        vector<int> final_token_chosen;
        vector<int> cur_chosen_pos_p1(cur_chosen_pos.begin(), cur_chosen_pos.begin() + v_s + 1);
        vector<int> cur_chosen_pos_p2(cur_chosen_pos.begin() + v_s + 1, cur_chosen_pos.begin() + v_e);
        vector<int> cur_chosen_pos_p3(cur_chosen_pos.begin() + v_e, cur_chosen_pos.end());
        final_token_chosen = cur_chosen_pos_p1;
        for (auto item : topk_indices) {
            final_token_chosen.push_back(cur_chosen_pos_p2[item]);
        }
        final_token_chosen.insert(final_token_chosen.end(), cur_chosen_pos_p3.begin(), cur_chosen_pos_p3.end());
        std::sort(final_token_chosen.begin(), final_token_chosen.end());
        return final_token_chosen;
    }
    vector<int> select_high_score_visual_token_decode(Tensor attn, int layer_idx, int chunk_size = 4) {
        auto cur_chosen_pos = chosen_pos_in_each[layer_idx];
        // attention_score_analyze_decode start
        attn = attn.mean(HEAD); // 1,t,1,t TODO
        if (attn.sequence() != 1) {
            attn = attn.clip({}, {}, {-1}, {}); // 1,1,1,t
        }
        auto cur_chosen_tokens = chosen_pos_in_each[layer_idx];
        int visual_start_in_selected = -1;
        int visual_end_in_selected = -1;
        for (int i = 0; i < cur_chosen_pos.size(); ++i) {
            auto pos = cur_chosen_pos[i];
            if (pos == first_img_token_pos - 1) { // <visual_start
                visual_start_in_selected = i;
            }
            if (pos == last_img_token_pos) { // <visual_end
                visual_end_in_selected = i;
            }
            if (visual_start_in_selected > 0 && visual_end_in_selected > 0) {
                break;
            }
        }
        int attn_dim_start = visual_start_in_selected + 1; // +1 for the first image token
        int attn_dim_end = visual_end_in_selected;         // exclusive
        vector<float> attn_score;                          // 1,1,1,visual_end_in_selected - visual_start_in_selected + 1
        for (int j = attn_dim_start; j < attn_dim_end; ++j) {
            float data = attn.dataAt<float>(0, 0, 0, j);
            attn_score.push_back(data);
        }
        auto v_s = visual_start_in_selected;
        auto v_e = visual_end_in_selected;
        // attention_score_analyze_decode end
        auto pruning_rate = pruning_place_cfg[layer_idx];
        auto cur_visual_token_length = attn_score.size();
        auto keep_ratio = 1 - pruning_rate;
        int k_initial = static_cast<int>(std::ceil(cur_visual_token_length * keep_ratio));
        int k_final = (k_initial / chunk_size) * chunk_size;
        k_final = std::min(k_final, static_cast<int>(cur_visual_token_length)); // 确保不超过当前实际长度
        vector<float> topk_vals;
        vector<int> topk_indices;
        topk_partial_sort(attn_score, k_final, topk_vals, topk_indices); // torch.topk(attn_score, k_final)
        vector<int> final_token_chosen;
        vector<int> cur_chosen_pos_p1(cur_chosen_pos.begin(), cur_chosen_pos.begin() + v_s + 1);
        vector<int> cur_chosen_pos_p2(cur_chosen_pos.begin() + v_s + 1, cur_chosen_pos.begin() + v_e);
        vector<int> cur_chosen_pos_p3(cur_chosen_pos.begin() + v_e, cur_chosen_pos.end());
        final_token_chosen = cur_chosen_pos_p1;
        for (auto item : topk_indices) {
            final_token_chosen.push_back(cur_chosen_pos_p2[item]);
        }
        final_token_chosen.insert(final_token_chosen.end(), cur_chosen_pos_p3.begin(), cur_chosen_pos_p3.end());
        std::sort(final_token_chosen.begin(), final_token_chosen.end());
        return final_token_chosen;
    }

    void update_hidden_pos(Tensor hidden_states, Tensor attn_weight, int layer_idx) {
        if (Module::llm_model_ptr->doLoad) { return; }
        if (is_prefill()) {
            auto chs_pos = chosen_pos_in_each[layer_idx];
            if (pruning_place_cfg.find(layer_idx) != pruning_place_cfg.end()) {
                kvcache_ctx.update_hidden_states(hidden_states, layer_idx, original_kv_length, chs_pos, is_prefill());
                chosen_pos_in_each[layer_idx + 1] = select_high_score_visual_token_prefill(attn_weight, layer_idx, chunk_size);
            } else {
                if (layer_idx + 1 < num_hidden_layers) {
                    chosen_pos_in_each[layer_idx + 1] = chosen_pos_in_each[layer_idx];
                }
            }
        } else {
            auto chs_pos = chosen_pos_to_delay_compute[layer_idx];
            if (pruning_place_cfg.find(layer_idx) != pruning_place_cfg.end()) {
                kvcache_ctx.update_hidden_states(hidden_states, layer_idx, original_kv_length, chs_pos, is_prefill());
                chosen_pos_in_each[layer_idx + 1] = select_high_score_visual_token_decode(attn_weight, layer_idx, chunk_size);
            } else {
                if (layer_idx + 1 < num_hidden_layers) {
                    chosen_pos_in_each[layer_idx + 1] = chosen_pos_in_each[layer_idx];
                }
            }
        }
    }

    Tensor prepare_next_layer(int layer_idx, Tensor &position_ids, Tensor &hidden_states, int kv_seq_len) {
        if (Module::llm_model_ptr->doLoad) { return hidden_states; }
        if (is_prefill()) {
            if (pruning_place_cfg.find(layer_idx) != pruning_place_cfg.end()) {
                auto this_layer_pos = chosen_pos_in_each[layer_idx];
                auto next_layer_pos = chosen_pos_in_each[layer_idx + 1];
                position_ids = global_position_ids.clip(next_layer_pos, DIMENSION);
                std::vector<int> mapping_this_2_next_pos;
                for (size_t idx = 0; idx < this_layer_pos.size(); ++idx) {
                    int value = this_layer_pos[idx];
                    if (std::find(next_layer_pos.begin(), next_layer_pos.end(), value) != next_layer_pos.end()) {
                        mapping_this_2_next_pos.push_back(idx);
                    }
                }
                assert(mapping_this_2_next_pos.size() == next_layer_pos.size());
                hidden_states = hidden_states.clip(mapping_this_2_next_pos, SEQUENCE);
            } else {
                if (layer_idx + 1 < num_hidden_layers) {
                    auto this_layer_pos = chosen_pos_in_each[layer_idx];
                    auto next_layer_pos = chosen_pos_in_each[layer_idx + 1];
                    assert(this_layer_pos.size() == next_layer_pos.size());
                    assert(std::equal(this_layer_pos.begin(), this_layer_pos.end(), next_layer_pos.begin()));
                }
            }
        } else {
            if (pruning_place_cfg.find(layer_idx) != pruning_place_cfg.end()) {
                auto this_layer_pos = chosen_pos_in_each[layer_idx];
                auto next_layer_pos = chosen_pos_in_each[layer_idx + 1];
                auto next_layer_kv_cache_not_filled_pos = kvcache_ctx.kv_not_filled_pos(layer_idx + 1, original_kv_length);
                std::vector<int> need_to_delay_compute_in_next_layer_pos;
                for (int item : next_layer_pos) {
                    if (std::find(next_layer_kv_cache_not_filled_pos.begin(),
                                  next_layer_kv_cache_not_filled_pos.end(),
                                  item)
                        != next_layer_kv_cache_not_filled_pos.end()) {
                        need_to_delay_compute_in_next_layer_pos.push_back(item);
                    }
                }
                std::sort(need_to_delay_compute_in_next_layer_pos.begin(),
                          need_to_delay_compute_in_next_layer_pos.end());
                chosen_pos_to_delay_compute[layer_idx + 1] = need_to_delay_compute_in_next_layer_pos;
                if (!need_to_delay_compute_in_next_layer_pos.empty()) {
                    position_ids = Tensor::cat(
                        {global_position_ids.clip(need_to_delay_compute_in_next_layer_pos, DIMENSION),
                         position_ids.clip({}, {}, {}, {-1})},
                        DIMENSION);
                    hidden_states = kvcache_ctx.reset_hidden_states(hidden_states, layer_idx, need_to_delay_compute_in_next_layer_pos);
                    // mask
                    int seq = chosen_pos_to_delay_compute[layer_idx + 1].size();
                    int dim = kv_seq_len + hidden_states.sequence();
                    auto &delay_compute_vec = chosen_pos_to_delay_compute[layer_idx + 1];
                    auto &in_each_vec = chosen_pos_in_each[layer_idx + 1];
                    Tensor causal_mask(1, num_head, seq + 1, dim, MLLM_CPU, true);
                    causal_mask.setName("causal_mask_" + std::to_string(layer_idx + 1));
                    float min_val = std::numeric_limits<float>::lowest();
                    for (int q_side_idx = 0; q_side_idx < seq; ++q_side_idx) {
                        // 获取当前查询位置对应的值
                        int target_value = delay_compute_vec[q_side_idx];
                        // 在in_each_vec中查找target_value的位置
                        auto it = std::find(in_each_vec.begin(), in_each_vec.end(), target_value);
                        // 确保找到目标值
                        if (it == in_each_vec.end()) {
                            // 处理未找到的情况 - 可选择报错或跳过
                            std::cerr << "Error: target_value not found in chosen_pos_in_each" << std::endl;
                            continue; // 跳过当前迭代
                        }
                        // 计算在向量中的索引位置
                        int start_index = std::distance(in_each_vec.begin(), it) + 1;
                        // 设置从start_index到末尾的所有元素为min_val
                        for (int h = 0; h < num_head; h++) {
                            for (int j = 0; j < start_index; ++j) {
                                causal_mask.setDataAt<float>(0, h, q_side_idx, j, 0);
                            }
                            for (int j = start_index; j < dim; ++j) {
                                causal_mask.setDataAt<float>(0, h, q_side_idx, j, -INFINITY);
                            }
                        }
                    }

                    for (int h = 0; h < num_head; h++) {
                        memset(causal_mask.ptrAt<float>(0, h, causal_mask.sequence() - 1, 0),
                               0, causal_mask.dimension() * sizeof(float));
                    }
                    causal_masks[layer_idx + 1] = causal_mask;
                } else {
                    hidden_states = hidden_states.clip({}, {}, {-1}, {});
                    position_ids = position_ids.clip({}, {}, {}, {-1});
                }
            } else {
                if (layer_idx + 1 < num_hidden_layers) {
                    auto this_layer_pos = chosen_pos_in_each[layer_idx];
                    auto next_layer_pos = chosen_pos_in_each[layer_idx + 1];
                    chosen_pos_to_delay_compute[layer_idx + 1] = chosen_pos_to_delay_compute[layer_idx];
                }
            }
        }
        return hidden_states;
    }
};
NdcContext WHERE_TOKEN_PRUNING;

#endif // NDC_TOOLS_HPP