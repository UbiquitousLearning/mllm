//
// Created by Rongjie Yi on 24-6-20.
//

#ifndef MODELING_MINICPM_HPP
#define MODELING_MINICPM_HPP

#include "Types.hpp"
#include "configuration_minicpm3.hpp"
#include <string>

using namespace mllm;

class MiniCPM3MultiHeadLatentAttention final : public Module {
    int hidden_size = 0;
    int num_heads = 0;
    int max_position_embeddings = 0;
    float rope_theta = 0.f;
    int q_lora_rank = 0;
    int qk_rope_head_dim = 0;
    int kv_lora_rank = 0;
    int v_head_dim = 0;
    int qk_nope_head_dim = 0;
    int q_head_dim = 0;

    Layer q_a_proj;
    Layer q_a_layernorm;
    Layer q_b_proj;
    Layer kv_a_proj_with_mqa;
    Layer kv_a_layernorm;
    Layer kv_b_proj;
    Layer o_proj;
    Layer q_rope;
    Layer k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;

    float softmax_scale = 0.f;

public:
    MiniCPM3MultiHeadLatentAttention() = default;
    MiniCPM3MultiHeadLatentAttention(const MiniCPM3Config config, const MiniCPM3NameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        max_position_embeddings = config.max_position_embeddings;
        rope_theta = config.rope_theta;
        q_lora_rank = config.q_lora_rank;
        qk_rope_head_dim = config.qk_rope_head_dim;
        kv_lora_rank = config.kv_lora_rank;
        v_head_dim = config.hidden_size / config.num_attention_heads;
        qk_nope_head_dim = config.qk_nope_head_dim;
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;

        q_a_proj = Linear(
            hidden_size,
            q_lora_rank,
            false,
            base_name + names._q_a_proj_name);

        q_a_layernorm = RMSNorm(
            q_lora_rank,
            config.rms_norm_eps,
            base_name + names._q_a_layernorm);

        q_b_proj = Linear(
            q_lora_rank,
            num_heads * q_head_dim,
            false,
            base_name + names._q_b_proj_name);

        kv_a_proj_with_mqa = Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            false,
            base_name + names._kv_a_proj_with_mqa_name);

        kv_a_layernorm = RMSNorm(
            kv_lora_rank,
            config.rms_norm_eps,
            base_name + names._kv_a_layernorm_name);

        kv_b_proj = Linear(
            kv_lora_rank,
            num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim),
            false,
            base_name + names._kv_b_proj_name);

        o_proj = Linear(
            num_heads * v_head_dim,
            hidden_size,
            false,
            base_name + names._o_proj_name);

        q_rope = NTKRoPE(
            HFHUBROPE,
            rope_theta,
            max_position_embeddings,
            config.rope_original_max_position_embeddings,
            config.rope_long_factor,
            config.rope_short_factor,
            base_name + "q_rope");

        k_rope = NTKRoPE(
            HFHUBROPE,
            rope_theta,
            max_position_embeddings,
            config.rope_original_max_position_embeddings,
            config.rope_long_factor,
            config.rope_short_factor,
            base_name + "k_rope");

        if (config.cache_limit > 0) {
            k_cache = KVCache(num_heads / num_heads, config.cache_limit, base_name + "k_cache");
            v_cache = KVCache(num_heads / num_heads, config.cache_limit, base_name + "v_cache");
        }

        softmax = Softmax(DIMENSION, config.do_mask, base_name + "softmax");

        softmax_scale = 1.f / (float)std::sqrt(q_head_dim);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = inputs[0];

        auto bsz = hidden_states.batch();
        auto q_len = hidden_states.sequence();

        // q: [bs, len, 1, num_heads * q_head_dim]
        auto q = q_b_proj(q_a_layernorm(q_a_proj(hidden_states)));
        // q_nope: [bs, len, num_heads, qk_nope_head_dim]
        // q_pe: [bs, len, num_heads, qk_rope_head_dim]
        auto qs = q.split({qk_nope_head_dim, qk_rope_head_dim}, D_HD, num_heads);
        auto q_nope = qs[0];
        auto q_pe = q_rope(qs[1]);
        auto query_states = Tensor::cat({q_nope, q_pe}, DIMENSION);

        // compressed_kv: [bs, len, 1, kv_lora_rank + qk_rope_head_dim]
        auto compressed_kv = kv_a_proj_with_mqa(hidden_states);
        // compressed_kv: [bs, len, 1, kv_lora_rank]
        // k_pe: [bs, len, 1, qk_rope_head_dim]
        auto kvs = compressed_kv.split({kv_lora_rank, qk_rope_head_dim}, DIMENSION);
        compressed_kv = kvs[0];
        Tensor k_pe = kvs[1];
        // kv: [bs, len, 1, num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)]
        auto kv = kv_b_proj(kv_a_layernorm(compressed_kv));
        // k_nope: [bs, len, num_heads, qk_nope_head_dim]
        // value_states: [bs, len, num_heads, v_head_dim]
        kvs = kv.split({qk_nope_head_dim, v_head_dim}, D_HD, num_heads);
        Tensor k_nope = kvs[0];
        Tensor value_states = kvs[1];
        k_pe = k_rope(k_pe);
        // k_pe = Tensor::cat(std::vector<Tensor>(num_heads, k_pe), HEAD); //没用，已经和下一个cat算子合并了
        auto key_states = Tensor::cat({k_nope, k_pe}, DIMENSION);

        // attention
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);
        key_states = key_states.transpose(SEQUENCE, DIMENSION);
        auto attn_weight = Tensor::mm(query_states, key_states);
        attn_weight = attn_weight * softmax_scale;
        attn_weight = softmax(attn_weight, k_cache.getCacheSeqLen());
        auto attn_output = Tensor::mm(attn_weight, value_states);
        attn_output = attn_output.view(-1, 1, -1, v_head_dim * num_heads);
        attn_output = o_proj(attn_output);
        return {attn_output};
    }
};

class MiniCPM3MLP final : public Module {
private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer silu;

public:
    MiniCPM3MLP() = default;
    MiniCPM3MLP(const MiniCPM3Config &config, const MiniCPM3NameConfig &names, const std::string &base_name) {
        int hidden_size = config.hidden_size;
        int intermediate_size = config.intermediate_size;

        gate_proj = Linear(
            hidden_size,
            intermediate_size,
            false,
            base_name + names._gate_proj_name);

        silu = SiLU(base_name + "act");

        up_proj = Linear(
            hidden_size,
            intermediate_size,
            false,
            base_name + names._up_proj_name);

        down_proj = Linear(
            intermediate_size,
            hidden_size,
            false,
            base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        return {down_proj(silu(gate_proj(x)) * up_proj(x))};
    }
};

class MiniCPM3Decoder final : public Module {
private:
    MiniCPM3MultiHeadLatentAttention self_attn;
    MiniCPM3MLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    float scale;

public:
    MiniCPM3Decoder() = default;
    MiniCPM3Decoder(const MiniCPM3Config &config, const MiniCPM3NameConfig &names, const string &base_name) {
        self_attn = MiniCPM3MultiHeadLatentAttention(
            config,
            names,
            base_name + names._attn_base_name);

        mlp = MiniCPM3MLP(
            config,
            names,
            base_name + names._ffn_base_name);

        input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            base_name + names._attn_norm_name);

        post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            base_name + names._ffn_norm_name);
        
        scale = config.scale_depth / std::sqrt(config.num_hidden_layers);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = self_attn({x, x, x})[0];
        auto residual = x*scale + inputs[0];
        x = post_attention_layernorm(residual);
        x = mlp({x})[0];
        x = x*scale + residual;
        return {x};
    }
};

class MiniCPM3Model final : public Module {
private:
    std::vector<MiniCPM3Decoder> blocks;
    Layer norm;

public:
    MiniCPM3Model() = default;
    MiniCPM3Model(const MiniCPM3Config &config, const MiniCPM3NameConfig &names, const string &base_name) {
        blocks = List<MiniCPM3Decoder>(
            config.num_hidden_layers,
            config,
            names,
            base_name);

        norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            true,
            names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        return {x};
    }
};

class MiniCPM3ForCausalLM final : public Module {
private:
    int hidden_size;
    Layer embedding;
    Parameter lm_head;
    MiniCPM3Model model;
    float scale_emb;

public:
    explicit MiniCPM3ForCausalLM(MiniCPM3Config &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;

        embedding = Embedding(
            config.vocab_size,
            config.hidden_size,
            names.token_embd_name);

        model = MiniCPM3Model(
            config,
            names,
            names.blk_name);

        // lm_head and tok_embedding is tied together.
        // They share same parameters. Use a Transpose to do the lm_head instead.
        lm_head = Parameter(
            1,
            config.vocab_size,
            1,
            config.hidden_size,
            names.lm_head_name + ".weight");

        scale_emb = config.scale_emb;
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0])* scale_emb;
        auto outputs = model({x})[0];
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {outputs};
    }
};

#endif // MODELING_MINICPM_HPP
