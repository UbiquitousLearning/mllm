//
// Created by Rongjie Yi on 25-3-29.
//
#ifndef VTP_TOOLS_HPP
#define VTP_TOOLS_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
// #include "../configuration_qwen2_vl.hpp"
// #include "models/qwen/modeling_qwen.hpp"
// #include <cassert>
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

using namespace mllm;

class VtpContext {
public:
    void init() {
        if (global_selected.backend() == nullptr)
            global_selected = Tensor(1, 1, 1, 1, MLLM_CPU);
    }
    void set_vision_token(Tensor where_idx, Tensor hidden_states, Tensor image_embeds) {
        // if (Module::llm_model_ptr->doLoad) {
        //     return;
        // }
        no_visual_token_len = hidden_states.sequence() - image_embeds.sequence();
        global_selected.reshape(1, 1, 1, hidden_states.sequence()); // pre_visual_token_len);
        global_selected.alloc();
        for (int i = 0; i < hidden_states.sequence(); ++i) {
            global_selected.setDataAt<float>(0, 0, 0, i, i);
        }
        first_img_token_pos = int(where_idx.dataAt<float>(0, 0, 0, 0));
        last_img_token_pos = int(where_idx.dataAt<float>(0, 0, 0, where_idx.dimension() - 1));
        pre_visual_token_len = last_img_token_pos - first_img_token_pos + 1;
        static_first_img_token_pos = first_img_token_pos;
        static_last_img_token_pos = last_img_token_pos;
        before_prefill_visual_tokens = pre_visual_token_len;
        no_visual_token_len = hidden_states.sequence() - pre_visual_token_len;
    }
    bool is_prefill() {
        // if (Module::llm_model_ptr->doLoad) {
        //     return false;
        // }
        return prefill_stage;
    }
    void set_prefill_layer(int layer_idx_) {
        if (prefill_stage)
            layer_idx = layer_idx_;
    }
    Tensor pruning_pos(Tensor input, Chl dim = Chl::SEQUENCE) {
        if (pruning_setting.find(layer_idx - 1) != pruning_setting.end()) {
            if (layer_idx - 1 == 0) {
                return input;
            }
            return input.clip(global_selected, dim);
        }
        return input;
    }
    Tensor pruning_(Tensor input, Chl dim = Chl::SEQUENCE) {
        if (pruning_setting.find(layer_idx) != pruning_setting.end()) {
            if (layer_idx == 0) {
                return input;
            }
            return input.clip(global_selected, dim);
        }
        return input;
    }
    void update_attn_acc_score(Tensor attn_weight) {
        if (Module::llm_model_ptr->doLoad) {
            return;
        }
        int visual_len;
        if (layer_idx == 0) {
            visual_len = last_img_token_pos - first_img_token_pos;
        } else {
            visual_len = global_selected.dimension() - 1 - no_visual_token_len;
        }
        // 计算切片参数
        const int total_heads = attn_weight.head();
        const int total_rows = attn_weight.sequence();
        const int total_cols = attn_weight.sequence();
        const int start_row = first_img_token_pos + visual_len + 1;
        const int num_rows = std::max(0, total_rows - start_row);
        const int num_cols = std::max(0, std::min(visual_len + 1, total_cols - first_img_token_pos));
        // 阶段1: 计算带转置的注意力求和矩阵
        vector<vector<float>> sum_matrix(num_rows, vector<float>(total_heads, 0));
        for (int h = 0; h < total_heads; ++h) {
            for (int r = 0; r < num_rows; ++r) {
                const int src_row = start_row + r;
                for (int c = 0; c < num_cols; ++c) {
                    sum_matrix[r][h] += attn_weight.dataAt<float>(0, h, src_row, first_img_token_pos + c);
                }
            }
        }
        // 阶段2: 生成排序索引
        vector<vector<int>> sorted_indices(num_rows);
        for (int r = 0; r < num_rows; ++r) {
            vector<std::pair<float, int>> heads;
            heads.reserve(total_heads);
            for (int h = 0; h < total_heads; ++h) {
                heads.emplace_back(sum_matrix[r][h], h);
            }
            // 仅排序前TOP_K元素
            nth_element(heads.begin(), heads.begin() + HEAD_TOP_K, heads.end(),
                        [](const auto &a, const auto &b) { return a.first > b.first; });
            // 提取有效头部索引
            for (int i = 0; i < std::min(HEAD_TOP_K, (int)heads.size()); ++i) {
                sorted_indices[r].push_back(heads[i].second);
            }
        }
        // 阶段3: 直接计算最终结果
        vector<float> attn_score(num_cols, 0.0f);
        const float norm = 1.0f / (HEAD_TOP_K * num_rows);
        for (int r = 0; r < num_rows; ++r) {
            const int valid_heads = std::min(HEAD_TOP_K, (int)sorted_indices[r].size());
            for (int i = 0; i < valid_heads; ++i) {
                const int h = sorted_indices[r][i];
                for (int c = 0; c < num_cols; ++c) {
                    attn_score[c] += attn_weight.dataAt<float>(0, h, start_row + r, first_img_token_pos + c) * norm;
                }
            }
        }

        if (gloabl_visual_attn_score.empty()) {
            gloabl_visual_attn_score = attn_score;
        } else {
            // auto &global_selected_p = Module::llm_model_ptr->activation_tensors["global_selected"];
            for (int d = 0; d < global_selected.dimension(); ++d) {
                auto data_i = global_selected.dataAt<float>(0, 0, 0, d);
                if (data_i >= static_first_img_token_pos && data_i <= static_last_img_token_pos) {
                    int i = data_i - static_first_img_token_pos;
                    gloabl_visual_attn_score[i] = ATTN_ACC_ALPHA * gloabl_visual_attn_score[i] + (1 - ATTN_ACC_ALPHA) * attn_score[i];
                }
            }
        }
    }
    Tensor prunning_attn_output(Tensor attn_output) {
        if (layer_idx == 0) {
            return attn_output;
        }

        // if (Module::llm_model_ptr->doLoad) {
        //     return attn_output;
        // }
        if (pruning_setting.find(layer_idx) != pruning_setting.end()) {
            auto cur_pruning_rate = pruning_setting[layer_idx];
            auto seq_len = attn_output.sequence() - no_visual_token_len;               // gloabl_visual_attn_score.size();
            auto k_num = int(cur_pruning_rate * seq_len);                              // static_cast<int>(std::ceil(cur_pruning_rate * seq_len));
            global_selected.reshape(1, 1, 1, (seq_len - k_num) + no_visual_token_len); // TODO
            global_selected.alloc();
            auto indices = topk_indices(gloabl_visual_attn_score, k_num); // vison中的idx
            std::vector<bool> mask(gloabl_visual_attn_score.size(), true);
            for (int idx : indices) {
                mask[idx] = false;
            }
            int iid = 0;
            size_t i = 0;
            for (; i < static_first_img_token_pos; ++i) {
                global_selected.setDataAt<float>(0, 0, 0, iid, i);
                iid++;
            }
            for (; i < mask.size() + static_first_img_token_pos; ++i) {
                if (mask[i - static_first_img_token_pos]) {
                    global_selected.setDataAt<float>(0, 0, 0, iid, i);
                    iid++;
                }
            }
            for (; i < mask.size() + no_visual_token_len; ++i) {
                global_selected.setDataAt<float>(0, 0, 0, iid, i);
                iid++;
            }
            last_img_token_pos = first_img_token_pos + seq_len - k_num - 1;
            return attn_output.clip(global_selected, SEQUENCE);
        } else {
            return attn_output;
        }
    }

    int first_img_token_pos = 0;
    int last_img_token_pos = 0;
    int pre_visual_token_len = 0;
    int no_visual_token_len = 0;
    int static_first_img_token_pos = 0;
    int static_last_img_token_pos = 0;
    int before_prefill_visual_tokens = 0;
    bool prefill_stage = true;
    int layer_idx;
    vector<float> gloabl_visual_attn_score;
    Tensor global_selected;
    int HEAD_TOP_K = 3;
    float ATTN_ACC_ALPHA = 0.2;

    map<int, float> pruning_setting = {{3, 0.5}}; //{{3, 0.5}};

private:
    // 实现 topk 功能
    std::vector<int> topk_indices(const std::vector<float> &scores, int k) {
        // 创建索引数组
        std::vector<int> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0); // 填充为 [0, 1, 2, ..., scores.size()-1]

        // 对索引数组进行排序，仅保留前 k 个最小值
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&scores](int i1, int i2) { return scores[i1] < scores[i2]; });

        // 返回前 k 个索引
        return std::vector<int>(indices.begin(), indices.begin() + k);
    }
};

VtpContext WHERE_TOKEN_PRUNING;

#endif // VTP_TOOLS_HPP