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
#include "Types.hpp"
#include "configuration_llama.hpp"
#include "backends/xnnpack/XpWrapper.hpp"

using namespace mllm;

// all in xnnpack
class XpLLaMAMHA final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer o_proj;
    Layer sdpa;

    int head_size_ = 0;
    int kv_head_size_ = 0;
    int attn_hidden_dim_ = 0;

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

        k_cache = XP_KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
        v_cache = XP_KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");

        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, false, base_name + names._o_proj_name);

        sdpa = ScaledDotProductAttention("sdpa");

        head_size_ = head_size;
        kv_head_size_ = kv_head_size;
        attn_hidden_dim_ = attn_hidden_dim;
    }

    vector<Tensor>
    Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // inputs is [B, S, H=1, D=dim]
        // Q, K, V is also [B, S, H=1, D=heads * dim]
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[0]);
        auto v = v_proj(inputs[0]);

        // q = q.view(bsz, q_len, num_heads, head_dim)
        // [B, S, H=heads, D=dim]
        q = q.view(-1, head_size_, -1, attn_hidden_dim_);
        k = k.view(-1, kv_head_size_, -1, attn_hidden_dim_);
        v = v.view(-1, kv_head_size_, -1, attn_hidden_dim_);

        q = q_rope(q);
        k = k_rope(k);

        k = k_cache(k);
        v = v_cache(v);

        // [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(SEQUENCE, HEAD);
        k = k.transpose(SEQUENCE, HEAD);
        v = v.transpose(SEQUENCE, HEAD);

        auto o = sdpa(q, k, v);

        // o is [B, H, S, D]
        // [B, H, S, D] -> [B, S, H, D]
        o = o.transpose(SEQUENCE, HEAD);
        // [B, S, H, D] -> [B, S, 1, H * D]
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);

        return {o};
    }
};

// all in xnnpack
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

// all in xnnpack
class LLaMABlock final : public Module {
    XpLLaMAMHA attention;
    LLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    LLaMABlock() = default;
    LLaMABlock(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, const LLaMANameConfig &names, const string &base_name) {
        attention = XpLLaMAMHA(hidden_dim, head_size, kv_head_size, hidden_dim / head_size,
                               RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name + names._attn_base_name);
        attention.to(BackendType::MLLM_XNNPACK);

        mlp = LLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        mlp.to(BackendType::MLLM_XNNPACK);

        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);
        x = attention({x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    XpLLaMAMHA &get_attention() {
        return attention;
    }
};

class LLaMAModelXp final : public Module {
    vector<LLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit LLaMAModelXp(const LLaMAConfig &config) :
        LLaMAModelXp(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                     config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                     config.names_config, config.names_config.blk_name) {
    }
    LLaMAModelXp(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
                 const LLaMANameConfig &names, const string &base_name) {
        blocks = List<LLaMABlock>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);

        for (auto &b : blocks) {
            b.to(BackendType::MLLM_XNNPACK);
        }

        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        // TODO bug here. one block is ok but fews
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

// all in xnnpack
class LLaMAModel final : public Module {
    Layer embedding;
    xnnpack::XpWrapperModule wrapped_model;

public:
    explicit LLaMAModel(const LLaMAConfig &config) :
        LLaMAModel(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                   config.names_config, config.names_config.blk_name) {
    }
    LLaMAModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
               const LLaMANameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        wrapped_model = xnnpack::wrap2xnn<LLaMAModelXp>(1, 1, vocab_size, hidden_dim, head_size, kv_head_size, ffn_hidden, block_num, RoPE_type, rope_theta, max_position_embeddings, cache_limit,
                                                        names, base_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        x = wrapped_model({x})[0];
        return {x};
    }

    // void clear_kvcache() {
    //     for (auto &block : blocks) {
    //         auto kvcahce = block.get_attention().get_cache();
    //         for (auto &cache : kvcahce) {
    //             cache->clearCache();
    //         }
    //     }
    // }
};