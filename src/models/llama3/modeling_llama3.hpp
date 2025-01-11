//
// Created by xwk on 25-1-10.
//

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama3.hpp"

using namespace mllm;

class Llama3MLP final : public Module {
    Layer gate_proj;
    Layer silu;
    Layer up_proj;
    Layer down_proj;

public:
    Llama3MLP() = default;
    Llama3MLP(int hidden_dim, int ffn_hidden, const Llama3NameConfig &names, const string &base_name) {
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

class Llama3Attention final : public Module {
    Layer q_proj;      // Query projection
    Layer k_proj;      // Key projection
    Layer v_proj;      // Value projection
    Layer o_proj;      // Output projection
    RoPE q_rope;       // RoPE for queries
    RoPE k_rope;       // RoPE for keys
    KVCache k_cache;   // Key cache
    KVCache v_cache;   // Value cache
    Softmax softmax;   // Softmax for attention scores
    int head_size_;    // Size of each attention head
    int kv_head_size_; // Size of each key/value head
    int hidden_dim_;   // Hidden dimension size

public:
    Llama3Attention() = default;

    Llama3Attention(int hidden_dim, int head_size, int kv_head_size, RoPEType RoPE_type, float rope_theta,
                    int max_position_embeddings, int cache_limit, const TransformerNameConfig &names,
                    const string &base_name, const RoPEConfig &rope_config = {}) {
        hidden_dim_ = hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;

        // Initialize projections
        q_proj = Linear(hidden_dim, head_size * (hidden_dim / head_size), false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_dim, kv_head_size * (hidden_dim / head_size), false, base_name + names._k_proj_name);
        v_proj = Linear(hidden_dim, kv_head_size * (hidden_dim / head_size), false, base_name + names._v_proj_name);
        o_proj = Linear(head_size * (hidden_dim / head_size), hidden_dim, false, base_name + names._o_proj_name);

        // Initialize RoPE
        if (!rope_config.empty()) {
            q_rope = RoPE(RoPE_type, rope_config, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, rope_config, base_name + "k_rope");
        } else if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "k_rope");
        }

        // Initialize KV cache
        if (cache_limit > 0) {
            k_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
            v_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");
        }

        // Initialize softmax
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor q = q_proj(inputs[0]); // Query projection
        Tensor k = k_proj(inputs[1]); // Key projection
        Tensor v = v_proj(inputs[2]); // Value projection

        // Reshape tensors for multi-head attention
        q = q.view(-1, head_size_, -1, hidden_dim_ / head_size_);
        k = k.view(-1, kv_head_size_, -1, hidden_dim_ / head_size_);
        v = v.view(-1, kv_head_size_, -1, hidden_dim_ / head_size_);

        // Apply RoPE
        if (q_rope.ready() && k_rope.ready()) {
            q = q_rope(q);
            k = k_rope(k);
        }

        // Update KV cache
        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }

        // Transpose keys for dot product
        k = k.transpose(SEQUENCE, DIMENSION);

        // Compute attention scores
        Tensor qk = Tensor::mm(q, k);                  // Dot product of queries and keys
        qk = qk / std::sqrt(hidden_dim_ / head_size_); // Scale by sqrt(d_k)

        // Apply softmax
        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen()); // Masked softmax if cache is used
        } else {
            qk = softmax(qk); // Regular softmax
        }

        // Compute attention output
        Tensor o = Tensor::mm(qk, v);       // Weighted sum of values
        o = o.view(-1, 1, -1, hidden_dim_); // Reshape to original dimensions
        o = o_proj(o);                      // Output projection

        return {o};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }

    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }
};

class Llama3Block final : public Module {
    Llama3Attention attention;
    Llama3MLP mlp;
    Layer norm1;
    Layer norm2;

public:
    Llama3Block() = default;
    Llama3Block(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
                const Llama3NameConfig &names,
                const Llama3Config &config,
                const string &base_name) {
        RoPEConfig rope_config;
        if (!config.rope_scaling.empty()) {
            rope_config["rope_theta"] = rope_theta;
            rope_config["max_position_embeddings"] = max_position_embeddings;
            rope_config["rope_scaling"] = config.rope_scaling;
        }

        attention = Llama3Attention(hidden_dim, head_size, kv_head_size, RoPE_type, rope_theta,
                                    max_position_embeddings, cache_limit, names, base_name + names._attn_base_name, rope_config);
        mlp = Llama3MLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
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

    Llama3Attention &get_attention() {
        return attention;
    }
};

class Llama3Model final : public Module {
    Layer embedding;
    vector<Llama3Block> blocks;
    Layer norm;
    Parameter lm_head;

public:
    explicit Llama3Model(const Llama3Config &config) :
        Llama3Model(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                    config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                    config.names_config, config, config.names_config.blk_name) {
    }
    Llama3Model(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
                const Llama3NameConfig &names,
                const Llama3Config &config,
                const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<Llama3Block>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, config, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        // TODO: tie_word_embeddings
        // this is a workaround
        // we just simply use the token embedding as the lm_head
        // but now we are not really tying the word embeddings
        auto lm_head_name = names.lm_head_name;
        if (config.tie_word_embeddings)
            lm_head_name = names.token_embd_name;
        lm_head = Parameter(1, vocab_size, 1, hidden_dim, lm_head_name + ".weight");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = Tensor::mm(x, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {x};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};
