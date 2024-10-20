/**
 * @file modeling_llama_xp_sdpa.hpp
 * @author your name (you@domain.com)
 * @version 0.1
 * @date 2024-10-20
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class XpLLaMAMHA final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer sdpa;
    Layer o_proj;

public:
    XpLLaMAMHA() = default;

    XpLLaMAMHA(
        int hidden_dim,
        int head_size,
        int kv_head_size,
        int attn_hidden_dim,
        RoPEType RoPE_type,
        float rope_theta,
        int max_position_embeddings,
        int cache_limit,
        const TransformerNameConfig &names,
        const string &base_name) {
        q_proj = Linear(hidden_dim, head_size * attn_hidden_dim, false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, false, base_name + names._k_proj_name);
        v_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, false, base_name + names._v_proj_name);

        q_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "k_rope");

        k_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
        v_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");

        sdpa = ScaledDotProductAttention(base_name + "sdpa");

        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, false, base_name + names._o_proj_name);
    }

    vector<Tensor>
    Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // inputs is [B, S, H, D]
        // Q, K, V is also [B, S, H, D]
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);

        // [B, S, H, D]
        q = q_rope(q);
        k = k_rope(k);

        // [B, S, H, D]
        k = k_cache(k);
        v = v_cache(v);

        // TODO Transpose Q,K,V to [B, H, S, D]
        auto o = sdpa(q, k, v);
        o = o_proj(o);

        return {o};
    }
};

class LLaMAMLP final : public Module {
    Layer gate_proj;
    Layer silu;
    Layer up_proj;
    Layer down_proj;

public:
    LLaMAMLP() = default;
    LLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name) {
        gate_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }
};

class LLaMABlock final : public Module {
    MultiHeadAttention attention;
    LLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    LLaMABlock() = default;
    LLaMABlock(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, const LLaMANameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size, hidden_dim / head_size, SPLIT_NONE, false, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, false, names, base_name + names._attn_base_name);

        mlp = LLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        mlp.to(BackendType::MLLM_XNNPACK);

        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    MultiHeadAttention &get_attention() {
        return attention;
    }
};

class LLaMAModel final : public Module {
    Layer embedding;
    vector<LLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit LLaMAModel(const LLaMAConfig &config) :
        LLaMAModel(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                   config.names_config, config.names_config.blk_name) {
    }
    LLaMAModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
               const LLaMANameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<LLaMABlock>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void clear_kvcache() {
        for (auto &block : blocks) {
            auto kvcahce = block.get_attention().get_cache();
            for (auto &cache : kvcahce) {
                cache->clearCache();
            }
        }
    }
};