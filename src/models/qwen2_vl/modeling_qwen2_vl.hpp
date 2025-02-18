//
// Created by Rongjie Yi on 25-2-9.
//
#ifndef MODELING_QWEN2VL_HPP
#define MODELING_QWEN2VL_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_qwen2_vl.hpp"
// #include "models/qwen/modeling_qwen.hpp"
#include <cassert>
#include <string>
#include <vector>

using namespace mllm;

class Qwen2PatchEmbed final : public Module {
    Layer proj;
    int embed_dim{};
public:
    Qwen2PatchEmbed() = default;
    Qwen2PatchEmbed(int vision_embed_dim, int patch, int img_hw, const Qwen2VLNameConfig &names, const string &base_name) {
        proj = Convolution3D(3, vision_embed_dim, {2, patch, patch}, {2, patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        embed_dim = vision_embed_dim;
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = proj(inputs[0]);
        embd = embd.view(1, 1, -1, embed_dim);
        return {embd};
    }
};

class VisionAttention final : public Module {
    Layer qkv_proj;
    Softmax softmax;
    Layer o_proj;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

public:
    VisionAttention() = default;
    VisionAttention(int hidden_dim, int head_size, int kv_head_size, int attn_hidden_dim, bool bias,
                       const TransformerNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;

        qkv_proj = Linear(hidden_dim, head_size * attn_hidden_dim * 3, bias, base_name + names._qkv_proj_name);
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, bias, base_name + names._o_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto cu_seqlens = inputs[1];
        auto rotary_pos_emb = inputs[2];
        auto seq_length = inputs[0].sequence();
        Tensor q, k, v;
        auto qkv = qkv_proj(inputs[0]);
        auto qkv_sp = qkv.split({attn_hidden_dim_, attn_hidden_dim_, attn_hidden_dim_}, HD, head_size_);
        q = qkv_sp[0];
        k = qkv_sp[1];
        v = qkv_sp[2];
        q = Tensor::apply_rotary_pos_emb_vision(q, rotary_pos_emb);
        k = Tensor::apply_rotary_pos_emb_vision(k, rotary_pos_emb);
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);
        //mask
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
};

class VisionMLP final : public Module {
    Layer up_proj;
    Layer act;
    Layer down_proj;

public:
    VisionMLP() = default;
    VisionMLP(int hidden_dim, int ffn_hidden, const string &act_fn_type, const ViTNameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, ffn_hidden, true, base_name + names._up_proj_name);
        act = ACT_FN[act_fn_type](base_name + names._ffn_base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, true, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        x = down_proj(x);
        return {x};
    }
};

class VisionBlock final : public Module {
    VisionAttention attention;
    VisionMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    VisionBlock() = default;
    VisionBlock(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type, const ViTNameConfig &names, const string &base_name) {
        attention = VisionAttention(hidden_dim, head_size, head_size, hidden_dim / head_size, true, names, base_name + names._attn_base_name);
        mlp = VisionMLP(hidden_dim, ffn_hidden, act_fn_type, names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto cu_seqlens = inputs[1];
        auto rotary_pos_emb = inputs[2];
        auto hidden_states = norm1(inputs[0]);
        hidden_states = attention({hidden_states, cu_seqlens, rotary_pos_emb})[0];
        auto residual = hidden_states + inputs[0];
        hidden_states = norm2(residual);
        hidden_states = mlp({hidden_states})[0];
        hidden_states = hidden_states + residual;
        return {hidden_states};
    }
};

class PatchMerger final : public Module {
    int hidden_size;
    Layer ln_q;
    Layer mlp0;
    Layer gelu;
    Layer mlp2;
public:
    PatchMerger() = default;
    PatchMerger(int dim, int context_dim, int spatial_merge_size, const Qwen2VLNameConfig &names, const string &base_name) {
        hidden_size = context_dim * (spatial_merge_size*spatial_merge_size);
        ln_q = LayerNorm(context_dim, true, 1e-6, base_name + names._ln_q_name);
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

class Qwen2VisionModel final : public Module {
    Qwen2PatchEmbed patch_embed;
    Layer rot_pos_emb;
    Layer pre_layrnorm;
    vector<VisionBlock> blocks;
    PatchMerger patch_merger;

public:
    Qwen2VisionModel() = default;
    Qwen2VisionModel(int hidden_dim, int vision_embed_dim, int head_size, int mlp_hidden_dim, const string &act_fn_type, int patch, int img_hw, int block_num, int spatial_merge_size, const Qwen2VLNameConfig &names, const string &base_name) {
        patch_embed = Qwen2PatchEmbed(vision_embed_dim, patch, img_hw, names, base_name + names.patch_embed_name);
        rot_pos_emb = VisionRoPE((vision_embed_dim/head_size)/2, spatial_merge_size, base_name + ".rot_pos_emb");
        blocks = List<VisionBlock>(block_num, vision_embed_dim, head_size, mlp_hidden_dim, act_fn_type, names, base_name + names._layer_name);
        patch_merger = PatchMerger(hidden_dim, vision_embed_dim, spatial_merge_size, names, base_name + names._merger_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = patch_embed({inputs[0]})[0];
        auto rotary_pos_emb = rot_pos_emb(inputs[1]);
        auto grid_t = inputs[0].dataAt<float>(0,0,0,0);
        auto grid_h = inputs[0].dataAt<float>(0,0,0,1);
        auto grid_w = inputs[0].dataAt<float>(0,0,0,2);
        vector<float> cu_seqlens_v = {0.0F, grid_t*grid_h*grid_w};
        auto cu_seqlens = Tensor(cu_seqlens_v);
        for (auto &block : blocks) {
            hidden_states = block({hidden_states, cu_seqlens, rotary_pos_emb})[0];
        }
        hidden_states = patch_merger({hidden_states})[0];
        return {hidden_states};
    }
};

class QWenMLP final : public Module {
public:
    QWenMLP() = default;
    QWenMLP(int hidden_size, int intermediate_size, const QWenNameConfig &names,
            const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj =
            Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer silu;
};

// Copied from GemmaAttention with Gemma->Qwen and using SWA
class QWenAttention final : public Module {
public:
    QWenAttention() = default;
    QWenAttention(const Qwen2VLConfig &config, const QWenNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true,
                        base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true,
                        base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        q_rope = MultimodalRoPE(config.rope_theta, config.max_position_embeddings, config.mrope_section, base_name + "q_rope");
        k_rope = MultimodalRoPE(config.rope_theta, config.max_position_embeddings, config.mrope_section, base_name + "k_rope");
        k_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "v_cache");
        // mask = SlidingWindowMask(config.sliding_window, base_name + "mask");
        mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);
        query_states = q_rope(query_states);
        key_states = k_rope(key_states);
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);
        auto atten_weight = 
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION))
            / std::sqrt(head_dim);
        atten_weight = mask(atten_weight, k_cache.getCacheSeqLen());
        atten_weight = softmax(atten_weight, k_cache.getCacheSeqLen());
        auto atten_output = Tensor::mm(atten_weight, value_states);
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        atten_output = o_proj(atten_output);
        return {atten_output};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<MultimodalRoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }

    private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    MultimodalRoPE q_rope;
    MultimodalRoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Causalmask mask;
    Softmax softmax;
};

// Copied from GemmaDecoder with Gemma->Qwen and set RmsNorm(without add_unit_offset)
class QWenDecoder final : public Module {
public:
    QWenDecoder() = default;
    QWenDecoder(const Qwen2VLConfig &config, const QWenNameConfig &names, const string &base_name) {
        self_atten = QWenAttention(config, names, base_name + names._attn_base_name);
        mlp = QWenMLP(config.hidden_size, config.intermediate_size, names,
                    base_name + names._ffn_base_name);
        input_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = self_atten({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = post_attention_layernorm(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
    QWenAttention &get_attention() {
        return self_atten;
    }

    private:
    QWenAttention self_atten;
    QWenMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};


class Qwen2VLModel final : public Module {
    Qwen2VisionModel visual;
    Layer embed_tokens;

    vector<QWenDecoder> blocks;
    Layer norm;
    Parameter lm_head;
    Layer lm_head_layer;

    int image_token_id;
    bool tie_embedding_words;

public:
    explicit Qwen2VLModel(const Qwen2VLConfig &config) {
        auto vocab_size = config.vocab_size;
        auto hidden_dim = config.hidden_size;
        auto head_size = config.num_attention_heads;
        auto ffn_hidden = config.intermediate_size;
        auto projection_cls = config.projection_cls;
        auto vision_embed_dim = config.vision_embed_dim;
        auto spatial_merge_size = config.spatial_merge_size;
        image_token_id = config.image_token_id;
        auto vision_names = config.vision_names_config;
        auto qwen_names = config.names_config;
        tie_embedding_words = config.tie_embedding_words;

        embed_tokens = Embedding(vocab_size, hidden_dim, qwen_names.token_embd_name);
        visual = Qwen2VisionModel(hidden_dim, vision_embed_dim, 16, vision_embed_dim * 4, "QuickGELU", 14, 336, 32, spatial_merge_size, vision_names, vision_names.vison_model_name);

        blocks = List<QWenDecoder>(config.num_hidden_layers, config, qwen_names, qwen_names.blk_name);
        norm = RMSNorm(hidden_dim, 1e-6, qwen_names.post_norm_name);
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size,qwen_names.token_embd_name + ".weight");
        } else {
            lm_head_layer = Linear(config.hidden_size, config.vocab_size, false, qwen_names.lm_head_name);
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        bool have_img = inputs[1].batch() > 0;
        auto hidden_states = embed_tokens({inputs[0]});
        if (have_img) {
            auto image_embeds = visual({inputs[1], inputs[2]})[0];
            auto n_image_features = image_embeds.sequence();
            auto where_idx = inputs[0].where(image_token_id, SEQUENCE);
            hidden_states = hidden_states.index_put(image_embeds, where_idx, false);
        }
        for (auto &block : blocks) {
            hidden_states = block({hidden_states})[0];
        }
        hidden_states = norm(hidden_states);
        if (tie_embedding_words) {
            hidden_states = Tensor::mm(hidden_states, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            hidden_states = lm_head_layer(hidden_states);
        }
        return {hidden_states};
    }
    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcahce = block.get_attention().get_cache();
            for (auto &cache : kvcahce) {
                cache->clearCache();
            }
        }
    }
};
#endif // MODELING_PHI3_HPP