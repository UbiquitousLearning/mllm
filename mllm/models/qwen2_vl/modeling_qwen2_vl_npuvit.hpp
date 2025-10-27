#ifndef MODELING_NPU_VIT_HPP
#define MODELING_NPU_VIT_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include "configuration_qwen2_vl.hpp"
#include <string>
#include <vector>

using namespace mllm;
namespace npu {

class VisionBlock_NPU final : public Module {
    Layer input_quantize;
    Layer qkv_dequant, q_view;
    Layer k_dequant, k_view;
    Layer v_dequant, v_view;

    Layer qkv_proj;
    Split qkv_split;
    Layer q_rope, k_rope;
    Layer pre_oproj_view;
    Layer o_proj;
    Layer o_quantize, post_oproj_dequantize;
    Layer qk_mm, qkv_mm;
    Softmax softmax;
    Layer scale;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

    Layer post_atten_res_add;

    Layer pre_mlp_quantize;
    Layer up_proj;
    Layer post_up_proj_dequantize;
    Layer act;
    Layer pre_down_proj_quantize;
    Layer down_proj;
    Layer post_down_proj_dequantize;

    Layer post_mlp_res_add;

    Layer norm1;
    Layer norm2;

public:
    VisionBlock_NPU() = default;
    VisionBlock_NPU(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type, const ViTNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = hidden_dim / head_size;
        head_size_ = head_size;
        kv_head_size_ = head_size;

        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name, true);

        // attention
        auto attn_base_name = base_name + names._attn_base_name;
        input_quantize = Quantize(true, attn_base_name + names._qkv_proj_name + ".quantize", MLLM_TYPE_I16);
        qkv_proj = Linear(hidden_dim, head_size * attn_hidden_dim_ * 3, false, attn_base_name + names._qkv_proj_name);
        // use FP16 for attention matmul
        qkv_dequant = DequantizeAdd(true, head_size * attn_hidden_dim_ * 3, attn_base_name + names._qkv_proj_name + ".dequantize", false, MLLM_TYPE_I16);

        qkv_split = Split(3, DIMENSION, head_size * attn_hidden_dim_, attn_base_name + names._qkv_proj_name + ".split");

        q_view = View(-1, 16, -1, attn_hidden_dim_, attn_base_name + names._qkv_proj_name + ".q-00_view_");
        k_view = View(-1, 16, -1, attn_hidden_dim_, attn_base_name + names._qkv_proj_name + ".k-00_view_");
        v_view = View(-1, 16, -1, attn_hidden_dim_, attn_base_name + names._qkv_proj_name + ".v-00_view_");

        q_rope = RoPESimple(-1, attn_base_name + "q_rope");
        k_rope = RoPESimple(-1, attn_base_name + "k_rope");

        softmax = Softmax(DIMENSION, false, attn_base_name + "softmax");

        qk_mm = Matmul(false, true, attn_base_name + "qk_mm");
        qkv_mm = Matmul(false, false, attn_base_name + "qkv_mm");
        scale = Scale(1 / std::sqrt(attn_hidden_dim_), 0, false, attn_base_name + "scale");

        pre_oproj_view = View(-1, 1, -1, attn_hidden_dim_ * head_size_, attn_base_name + "or_split-00_view_");

        o_quantize = Quantize(true, attn_base_name + names._o_proj_name + ".quantize", MLLM_TYPE_I16);
        o_proj = Linear(head_size * attn_hidden_dim_, hidden_dim, false, attn_base_name + names._o_proj_name);
        post_oproj_dequantize = DequantizeAdd(true, hidden_dim, attn_base_name + names._o_proj_name + ".dequantize", true, MLLM_TYPE_I16);

        post_atten_res_add = Add(attn_base_name + "post_atten_add");

        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name, true);

        // mlp
        auto mlp_base_name = base_name + names._ffn_base_name;
        pre_mlp_quantize = Quantize(true, mlp_base_name + names._up_proj_name + ".quantize", MLLM_TYPE_I16);
        up_proj = Linear(hidden_dim, ffn_hidden, false, mlp_base_name + names._up_proj_name);
        // NOTE: QNN GeLU doesn't support FP32, use FP16
        post_up_proj_dequantize = DequantizeAdd(true, ffn_hidden, mlp_base_name + names._up_proj_name + ".dequantize", false, MLLM_TYPE_I16);

        act = ACT_FN[act_fn_type](mlp_base_name + "act");

        pre_down_proj_quantize = Quantize(true, mlp_base_name + names._down_proj_name + ".quantize", MLLM_TYPE_I16);
        down_proj = Linear(ffn_hidden, hidden_dim, false, mlp_base_name + names._down_proj_name);
        post_down_proj_dequantize = DequantizeAdd(true, hidden_dim, mlp_base_name + names._down_proj_name + ".dequantize", true, MLLM_TYPE_I16);

        post_mlp_res_add = Add(mlp_base_name + "res_add");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto after_norm1 = norm1(inputs[0]);
        auto hidden_states = after_norm1;

        // attention
        auto rotary_pos_emb_sin = inputs[1];
        auto rotary_pos_emb_cos = inputs[2];

        Tensor q, k, v;
        hidden_states = input_quantize(hidden_states);
        auto int_qkv = qkv_proj(hidden_states);
        auto qkv = int_qkv;

        qkv = qkv_dequant(qkv);

        auto qkv_sp = qkv_split(qkv);

        q = qkv_sp[0];
        k = qkv_sp[1];
        v = qkv_sp[2];

        q = q_view(q);
        k = k_view(k);
        v = v_view(v);

        q = q_rope(q, rotary_pos_emb_sin, rotary_pos_emb_cos);
        k = k_rope(k, rotary_pos_emb_sin, rotary_pos_emb_cos);

        auto qk = qk_mm(q, k);
        qk = scale(qk);
        qk = softmax(qk);
        auto o = qkv_mm(qk, v);

        o = pre_oproj_view(o);

        o = o_quantize(o);
        hidden_states = o_proj(o);
        hidden_states = post_oproj_dequantize(hidden_states);

        auto residual = post_atten_res_add(hidden_states, inputs[0]);

        hidden_states = norm2(residual);

        // mlp
        hidden_states = pre_mlp_quantize(hidden_states);
        hidden_states = up_proj(hidden_states);
        hidden_states = post_up_proj_dequantize(hidden_states);

        hidden_states = act(hidden_states);

        hidden_states = pre_down_proj_quantize(hidden_states);
        hidden_states = down_proj(hidden_states);
        hidden_states = post_down_proj_dequantize(hidden_states);

        hidden_states = post_mlp_res_add(hidden_states, residual);

        return {hidden_states};
    }
};

class Qwen2PatchEmbedForNPU final : public Module {
    Layer proj;
    int embed_dim{};

public:
    Qwen2PatchEmbedForNPU() = default;
    Qwen2PatchEmbedForNPU(int vision_embed_dim, int patch, int img_hw, const Qwen2VLNameConfig &names, const string &base_name) {
        proj = Convolution3D(3, vision_embed_dim, {2, patch, patch}, {2, patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        embed_dim = vision_embed_dim;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = proj(inputs[0]);
        embd = embd.view(1, 1, -1, embed_dim);
        return {embd};
    }
};

class RotationPatchMerger final : public Module {
    int hidden_size;
    Layer ln_q;
    Layer mlp0;
    Layer gelu;
    Layer mlp2;

public:
    RotationPatchMerger() = default;
    RotationPatchMerger(int dim, int context_dim, int spatial_merge_size, const Qwen2VLNameConfig &names, const string &base_name) {
        hidden_size = context_dim * (spatial_merge_size * spatial_merge_size);
        ln_q = RMSNorm(context_dim, 1e-6, base_name + names._ln_q_name, true);
        mlp0 = Linear(hidden_size, hidden_size, true, base_name + names._m_mlp_0_name);
        gelu = GELU(base_name + ".gelu");
        mlp2 = Linear(hidden_size, dim, true, base_name + names._m_mlp_2_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        x = mlp2(gelu(mlp0(ln_q(x).view(1, 1, -1, hidden_size))));
        return {x};
    }
};

class Qwen2VisionModel_NPU : public Module {
    Qwen2PatchEmbedForNPU patch_embed;

    Layer rot_pos_emb, rot_pos_emb_sin, rot_pos_emb_cos;
    Layer pre_layrnorm;
    vector<VisionBlock_NPU> blocks;
    RotationPatchMerger patch_merger;

    SubgraphStart _SubgraphStart;
    SubgraphFinalize _SubgraphEnd;

public:
    Qwen2VisionModel_NPU() = default;
    Qwen2VisionModel_NPU(int hidden_dim, int vision_embed_dim, int head_size, int mlp_hidden_dim, const string &act_fn_type, int patch, int img_hw, int block_num, int spatial_merge_size, const Qwen2VLNameConfig &names, const string &base_name) {
        patch_embed = Qwen2PatchEmbedForNPU(vision_embed_dim, patch, img_hw, names, base_name + names.patch_embed_name);
        rot_pos_emb = VisionRoPE((vision_embed_dim / head_size) / 2, spatial_merge_size, base_name + ".rot_pos_emb");
        rot_pos_emb_sin = VisionRoPESin((vision_embed_dim / head_size) / 2, spatial_merge_size, base_name + ".rot_pos_emb_sin");
        rot_pos_emb_cos = VisionRoPECos((vision_embed_dim / head_size) / 2, spatial_merge_size, base_name + ".rot_pos_emb_cos");

        blocks = List<VisionBlock_NPU>(block_num, vision_embed_dim, head_size, mlp_hidden_dim, act_fn_type, names, base_name + names._layer_name);
        patch_merger = RotationPatchMerger(hidden_dim, vision_embed_dim, spatial_merge_size, names, base_name + names._merger_name);

        _SubgraphStart = SubgraphStart(base_name + "subgraph_start");
        _SubgraphEnd = SubgraphFinalize(base_name + "subgraph_end");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = patch_embed({inputs[0]})[0];

        auto rotary_pos_emb_sin = rot_pos_emb_sin(inputs[1]);
        auto rotary_pos_emb_cos = rot_pos_emb_cos(inputs[1]);

        _SubgraphStart({hidden_states, rotary_pos_emb_sin, rotary_pos_emb_cos});

        for (int i = 0; i < blocks.size(); i++) {
            hidden_states = blocks[i]({hidden_states, rotary_pos_emb_sin, rotary_pos_emb_cos})[0];
        }

        _SubgraphEnd({hidden_states});

        hidden_states = patch_merger({hidden_states})[0];

        return {hidden_states};
    }
};

class Qwen2VL_ImagePatchAndEmbedding final : public Module {
    Qwen2VisionModel_NPU visual;
    Layer embed_tokens;

    Layer norm;
    Parameter lm_head;
    Layer lm_head_layer;

    bool tie_embedding_words;

    int64_t spatial_merge_size;
    int64_t image_token_id;
    int64_t video_token_id;
    int64_t vision_start_token_id;

public:
    explicit Qwen2VL_ImagePatchAndEmbedding(const Qwen2VLConfig &config) {
        auto vocab_size = config.vocab_size;
        auto hidden_dim = config.hidden_size;
        auto head_size = config.num_attention_heads;
        auto ffn_hidden = config.intermediate_size;
        auto projection_cls = config.projection_cls;
        auto vision_embed_dim = config.vision_embed_dim;
        image_token_id = config.image_token_id;
        auto vision_names = config.vision_names_config;
        auto qwen_names = config.names_config;
        tie_embedding_words = config.tie_embedding_words;
        spatial_merge_size = config.spatial_merge_size;
        image_token_id = config.image_token_id;
        video_token_id = config.video_token_id;
        vision_start_token_id = config.vision_start_token_id;

        embed_tokens = Embedding(vocab_size, hidden_dim, qwen_names.token_embd_name);
        // NOTE: Use GELU for NPU Qwen2VL ViT. the QuickGELU is implemented using QNN 1.702*x*sigmoid(x), which is slow
        visual = Qwen2VisionModel_NPU(hidden_dim, vision_embed_dim, 16, vision_embed_dim * 4, "GELU", 14, 336, 32, spatial_merge_size, vision_names, vision_names.vison_model_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = embed_tokens({inputs[0]});

        auto image_embeds = visual({inputs[1], inputs[2]})[0];
        auto n_image_features = image_embeds.sequence();
        auto where_idx = inputs[0].where(image_token_id, SEQUENCE);
        hidden_states = hidden_states.index_put(image_embeds, where_idx, false);

        return {hidden_states};
    }

    // changed from get_position_ids in CPU Qwen2VL, enable padding
    // when prefilling, padding_to should be the max length of the input
    // when decoding, real_seq should be the real length of the input, thus get the correct position_ids for decoding
    void get_position_ids(vector<Tensor> &inputs, int padding_to = 0, int real_seq = 0) {
        if (inputs[0].sequence() > 1) {
            Tensor video_grid_thw(0, 0, 0, 0, MLLM_CPU, true);
            auto rope_indices = get_rope_index_cpp(inputs[0], inputs[2], video_grid_thw, padding_to);
            auto position = rope_indices[0];
            if (inputs.size() == 4) {
                inputs[3] = position;
            } else {
                inputs.push_back(position);
            }
        } else {
            auto &position_ids = inputs[3];
            auto last_pos = real_seq == 0 ? position_ids.dataAt<float>(0, 0, 0, position_ids.dimension() - 1) : real_seq - 1;
            position_ids.reshape(position_ids.batch(), 1, position_ids.sequence(), 1);
            for (int b = 0; b < position_ids.batch(); b++) {
                for (int s = 0; s < position_ids.sequence(); s++) {
                    position_ids.setDataAt<float>(b, 0, s, 0, last_pos + 1);
                }
            }
        }
    }

private:
    vector<Tensor> get_rope_index_cpp(
        Tensor input_ids,
        Tensor image_grid_thw,
        Tensor video_grid_thw,
        int padding_to = 0) {
        vector<vector<int64_t>> attention_mask;
        auto attention_mask_shape = input_ids.sequence();
        for (int b = 0; b < input_ids.batch(); b++) {
            attention_mask.emplace_back(attention_mask_shape, 1);
        }
        const size_t batch_size = input_ids.batch(); // input_ids.size();

        // NOTE: changed from original
        const size_t seq_len = batch_size > 0 ? (padding_to > input_ids.sequence() ? padding_to : input_ids.sequence()) : 0; // batch_size > 0 ? input_ids[0].size() : 0;

        Tensor position_ids(3, 1, batch_size, seq_len, Backend::global_backends[MLLM_CPU].get(), true);
        Tensor mrope_position_deltas(1, 1, 1, batch_size, Backend::global_backends[MLLM_CPU].get(), true);
        bool has_vision = (image_grid_thw.sequence() > 0) || (video_grid_thw.sequence() > 0); // image_grid_thw || video_grid_thw;
        if (!has_vision) {
            // Pure text case
            for (size_t i = 0; i < batch_size; ++i) {
                const auto &mask = !attention_mask.empty() ? attention_mask[i] : vector<int64_t>(seq_len, 1);
                vector<int64_t> positions;
                int64_t pos = 0;
                for (size_t j = 0; j < seq_len; ++j) {
                    if (mask[j] == 1) {
                        positions.push_back(pos++);
                    } else {
                        positions.push_back(1); // Will be overwritten by mask
                    }
                }
                for (int dim = 0; dim < 3; ++dim) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        position_ids.setDataAt<float>(dim, 0, i, j, (float)(mask[j] == 1 ? positions[j] : 1));
                    }
                }
                int64_t max_pos = pos - 1;
                mrope_position_deltas.setDataAt<float>(0, 0, 0, i, (float)((max_pos + 1) - static_cast<int64_t>(input_ids.sequence())));
            }
            position_ids.setName("position_ids");
            mrope_position_deltas.setName("mrope_position_deltas");
            return {position_ids, mrope_position_deltas};
        }
        // Process vision cases
        size_t image_idx = 0, video_idx = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            const auto &mask = !attention_mask.empty() ? attention_mask[i] : vector<int64_t>(seq_len, 1);
            // Extract valid tokens
            vector<int64_t> valid_tokens;
            for (size_t j = 0; j < input_ids.sequence(); ++j) {
                if (mask[j] == 1) valid_tokens.push_back((int)input_ids.dataAt<float>(i, 0, j, 0));
            }
            // Find vision start positions
            vector<size_t> vision_starts;
            vector<int64_t> vision_types;
            for (size_t j = 0; j < valid_tokens.size(); ++j) {
                if (valid_tokens[j] == vision_start_token_id && j + 1 < valid_tokens.size()) {
                    vision_starts.push_back(j);
                    vision_types.push_back(valid_tokens[j + 1]);
                }
            }
            int64_t image_count = count(vision_types.begin(), vision_types.end(), image_token_id);
            int64_t video_count = vision_types.size() - image_count;
            vector<vector<int64_t>> llm_positions(3);
            size_t st = 0;
            int64_t current_max = 0;
            int64_t remain_images = image_count;
            int64_t remain_videos = video_count;
            // Process each vision segment
            for (size_t vs = 0; vs < vision_starts.size(); ++vs) {
                // Find next vision token
                size_t ed_image = valid_tokens.size();
                size_t ed_video = valid_tokens.size();
                if (remain_images > 0) {
                    auto it = find(valid_tokens.begin() + st, valid_tokens.end(), image_token_id);
                    if (it != valid_tokens.end()) ed_image = it - valid_tokens.begin();
                }
                if (remain_videos > 0) {
                    auto it = find(valid_tokens.begin() + st, valid_tokens.end(), video_token_id);
                    if (it != valid_tokens.end()) ed_video = it - valid_tokens.begin();
                }
                size_t ed = min(ed_image, ed_video);
                if (ed == valid_tokens.size()) break;
                // Get grid parameters
                int64_t t, h, w;
                bool is_image = (ed == ed_image);
                if (is_image) {
                    t = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 0);
                    h = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 1);
                    w = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 2);
                    image_idx++;
                    remain_images--;
                } else {
                    t = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 0);
                    h = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 1);
                    w = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 2);
                    video_idx++;
                    remain_videos--;
                }
                // Calculate grid dimensions
                int64_t llm_grid_t = t;
                int64_t llm_grid_h = h / spatial_merge_size;
                int64_t llm_grid_w = w / spatial_merge_size;
                // Process text segment
                size_t text_len = ed - st;
                if (text_len > 0) {
                    int64_t start_idx = current_max;
                    for (int64_t k = 0; k < text_len; ++k) {
                        for (int dim = 0; dim < 3; ++dim) {
                            llm_positions[dim].push_back(start_idx + k);
                        }
                    }
                    current_max += text_len;
                }
                for (int64_t ti = 0; ti < llm_grid_t; ++ti) {
                    for (int64_t hi = 0; hi < llm_grid_h; ++hi) {
                        for (int64_t wi = 0; wi < llm_grid_w; ++wi) {
                            llm_positions[0].push_back(current_max + ti);
                            llm_positions[1].push_back(current_max + hi);
                            llm_positions[2].push_back(current_max + wi);
                        }
                    }
                }
                current_max = std::max({llm_positions[0][llm_positions[0].size() - 1],
                                        llm_positions[1][llm_positions[1].size() - 1],
                                        llm_positions[2][llm_positions[2].size() - 1]});
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
            }
            // Process remaining text
            if (st < valid_tokens.size()) {
                size_t text_len = valid_tokens.size() - st;
                int64_t st_idx = current_max + 1;
                for (int64_t k = 0; k < text_len; ++k) {
                    for (int dim = 0; dim < 3; ++dim) {
                        llm_positions[dim].push_back(st_idx + k);
                    }
                }
                current_max += text_len;
            }
            // Fill position_ids with valid positions
            size_t valid_idx = 0;
            for (size_t j = 0; j < seq_len; ++j) {
                if (mask[j] == 1) {
                    if (valid_idx < llm_positions[0].size()) {
                        position_ids.setDataAt<float>(0, 0, i, j, (float)llm_positions[0][valid_idx]);
                        position_ids.setDataAt<float>(1, 0, i, j, (float)llm_positions[1][valid_idx]);
                        position_ids.setDataAt<float>(2, 0, i, j, (float)llm_positions[2][valid_idx]);
                        valid_idx++;
                    }
                }
            }
            // Calculate delta
            int64_t max_pos = 0;
            for (const auto &dim : llm_positions) {
                for (auto val : dim) {
                    max_pos = max(max_pos, val);
                }
            }
            mrope_position_deltas.setDataAt<float>(0, 0, 0, i, (float)((max_pos + 1) - static_cast<int64_t>(input_ids.sequence())));
        }
        position_ids.setName("position_ids");
        mrope_position_deltas.setName("mrope_position_deltas");
        return {position_ids, mrope_position_deltas};
    }
};
} // namespace npu

#endif // MODELING_QWEN2VL_NPU_HPP