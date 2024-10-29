/**
 * @file modeling_QWen_xp_sdpa.hpp
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
#include "configuration_qwen.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"

using namespace mllm;

// all in xnnpack
class XpQWenMHA final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer o_proj;
    Layer sdpa;

    int hidden_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    int num_key_value_heads = 0;
    int num_key_value_groups = 0;

public:
    XpQWenMHA() = default;

    XpQWenMHA(
        const QWenConfig &config,
        const QWenNameConfig &names,
        const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._v_proj_name);

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "k_rope");

        k_cache = XP_KVCache(num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = XP_KVCache(num_key_value_groups, config.cache_limit, base_name + "v_cache");

        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);

        sdpa = ScaledDotProductAttention(base_name + "sdpa");
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
        q = q.view(-1, num_heads, -1, head_dim);
        k = k.view(-1, num_key_value_heads, -1, head_dim);
        v = v.view(-1, num_key_value_heads, -1, head_dim);

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
        o = o.view(-1, 1, -1, head_dim * num_heads);
        o = o_proj(o);

        return {o};
    }
};

// all in xnnpack
class QWenMLP final : public Module {
public:
    QWenMLP() = default;
    QWenMLP(int hidden_size, int intermediate_size, const QWenNameConfig &names,
            const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
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

// all in xnnpack
class QWenDecoder final : public Module {
public:
    QWenDecoder() = default;
    QWenDecoder(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        self_atten = XpQWenMHA(config, names, base_name + names._attn_base_name);
        self_atten.to(BackendType::MLLM_XNNPACK);

        mlp = QWenMLP(config.hidden_size, config.intermediate_size, names,
                      base_name + names._ffn_base_name);
        mlp.to(BackendType::MLLM_XNNPACK);

        input_layernorm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, base_name + names._ffn_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = self_atten({x})[0];
        auto tmp = x + inputs[0];
        x = post_attention_layernorm(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    XpQWenMHA &get_attention() {
        return self_atten;
    }

private:
    XpQWenMHA self_atten;
    QWenMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

// all in xnn
class QWenModel_0_5 final : public Module {
public:
    QWenModel_0_5() = default;
    QWenModel_0_5(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        blocks = List<QWenDecoder>(config.num_hidden_layers, config, names, base_name);
        for (auto &b : blocks) b.to(BackendType::MLLM_XNNPACK);

        norm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, names.post_norm_name);

        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) { x = block({x})[0]; }
        x = norm(x);

        // x is [B, S, H ,D]
        // lm_head() is [1, Vocab, 1, D]
        // TODO Bug, transpose S and D will output [1, D, 1, Vaocb]. the third dim is 1!!!.
        x = Tensor::mm(x, lm_head().transpose(SEQUENCE, DIMENSION));
        return {x};
    }

private:
    std::vector<QWenDecoder> blocks;
    Layer norm;
    Parameter lm_head;
};

class QWenModel final : public Module {
public:
    QWenModel() = default;
    QWenModel(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        blocks = List<QWenDecoder>(config.num_hidden_layers, config, names, base_name);
        for (auto &b : blocks) b.to(BackendType::MLLM_XNNPACK);

        norm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, names.post_norm_name);

        lm_head = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) { x = block({x})[0]; }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

private:
    std::vector<QWenDecoder> blocks;
    Layer norm;
    Layer lm_head;
};

// all in xnnpack
class QWenForCausalLM final : public Module {
public:
    explicit QWenForCausalLM(QWenConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);

        // Qwen-0.5 use tied embedding
        // Others use nn.Linear()
        if (tie_embedding_words) {
            model = xnnpack::wrap2xnn<QWenModel_0_5>(1, 1, config, names, names.blk_name);
        } else {
            model = xnnpack::wrap2xnn<QWenModel>(1, 1, config, names, names.blk_name);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // go through model
        auto outputs = model({x})[0];

        return {outputs};
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    xnnpack::XpWrapperModule model;
};
