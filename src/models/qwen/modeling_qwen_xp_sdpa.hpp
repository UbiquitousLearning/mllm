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
#include "backends/xnnpack/Utils/Logger.hpp"

using namespace mllm;

// input_layer_norm + qkv_proj
class XpDecoderSeperatedPart_1_Xnn : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    RoPE q_rope;
    RoPE k_rope;
    Layer input_layernorm;

    int hidden_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    int num_key_value_heads = 0;
    int num_key_value_groups = 0;

public:
    XpDecoderSeperatedPart_1_Xnn() = default;
    XpDecoderSeperatedPart_1_Xnn(
        const QWenConfig &config,
        const QWenNameConfig &names,
        const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._attn_base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._attn_base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._attn_base_name + names._v_proj_name);

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + names._attn_base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + names._attn_base_name + "k_rope");

        input_layernorm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, base_name + names._attn_norm_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto normed_x = input_layernorm(inputs[0]);

        // inputs is [B, S, H=1, D=dim]
        // Q, K, V is also [B, S, H=1, D=heads * dim]
        auto q = q_proj(normed_x);
        auto k = k_proj(normed_x);
        auto v = v_proj(normed_x);

        // q = q.view(bsz, q_len, num_heads, head_dim)
        // [B, S, H=heads, D=dim]
        q = q.view(-1, num_heads, -1, head_dim);
        k = k.view(-1, num_key_value_heads, -1, head_dim);
        v = v.view(-1, num_key_value_heads, -1, head_dim);

        q = q_rope(q);
        k = k_rope(k);

        return {q, k, v};
    };
};

// kvcache
class XpDecoderSeperatedPart_2_Cpu : public Module {
    Layer k_cache;
    Layer v_cache;

    int hidden_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    int num_key_value_heads = 0;
    int num_key_value_groups = 0;

public:
    XpDecoderSeperatedPart_2_Cpu() = default;

    XpDecoderSeperatedPart_2_Cpu(
        const QWenConfig &config,
        const QWenNameConfig &names,
        const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        k_cache = XP_KVCache(num_key_value_groups, config.cache_limit, base_name + names._attn_base_name + "k_cache");
        v_cache = XP_KVCache(num_key_value_groups, config.cache_limit, base_name + names._attn_base_name + "v_cache");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto k = inputs[0];
        auto v = inputs[1];

        k = k_cache(k);
        v = v_cache(v);

        return {k, v};
    }
};

// sdpa + o_proj and mlp + post_norm.
class XpDecoderSeperatedPart_3_Xnn : public Module {
    Layer o_proj;
    Layer sdpa;

    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer silu;

    Layer post_attention_layernorm;

    int hidden_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    int num_key_value_heads = 0;
    int num_key_value_groups = 0;

public:
    XpDecoderSeperatedPart_3_Xnn() = default;

    XpDecoderSeperatedPart_3_Xnn(
        const QWenConfig &config,
        const QWenNameConfig &names,
        const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._attn_base_name + names._o_proj_name);
        sdpa = ScaledDotProductAttention(base_name + names._attn_base_name + "sdpa");

        post_attention_layernorm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, base_name + names._ffn_norm_name);

        gate_proj = Linear(hidden_size, config.intermediate_size, false, base_name + names._ffn_base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, config.intermediate_size, false, base_name + names._ffn_base_name + names._up_proj_name);
        down_proj = Linear(config.intermediate_size, hidden_size, false, base_name + names._ffn_base_name + names._down_proj_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];
        auto skip_connect = inputs[3];

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

        // skip connect
        auto tmp = o + skip_connect;
        auto in_mlp = post_attention_layernorm(tmp);

        // MLP Part
        auto x = gate_proj(in_mlp);
        x = silu(x);
        auto y = up_proj(in_mlp);
        x = x * y;
        x = down_proj(x);

        // skip connect
        x = x + tmp;

        return {x};
    }
};

// all in xnnpack
class QWenDecoder final : public Module {
public:
    QWenDecoder() = default;
    QWenDecoder(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        part_1 = XpDecoderSeperatedPart_1_Xnn(config, names, base_name);
        part_1.to(MLLM_XNNPACK);

        part_2 = XpDecoderSeperatedPart_2_Cpu(config, names, base_name);
        part_2.to(MLLM_CPU);

        part_3 = XpDecoderSeperatedPart_3_Xnn(config, names, base_name);
        part_3.to(MLLM_XNNPACK);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        // xnn side
        auto o_p_1 = part_1({inputs[0]}); // return q, k, v;
        o_p_1[1].to(MLLM_CPU);            // k
        o_p_1[2].to(MLLM_CPU);            // v

        // cpu side
        auto o_p_2 = part_2({o_p_1[1], o_p_1[2]}); // eats k, v. return k, v.
        o_p_2.insert(o_p_2.begin(), o_p_1[0]);
        o_p_2.push_back(inputs[0]);
        o_p_2[1].to(MLLM_XNNPACK); // k
        o_p_2[2].to(MLLM_XNNPACK); // v
        o_p_2[3].to(MLLM_XNNPACK); // skip connect

        // xnn side
        auto o = part_3(o_p_2)[0]; // MLP + skip connect.
        return {o};
    }

private:
    XpDecoderSeperatedPart_1_Xnn part_1;
    XpDecoderSeperatedPart_2_Cpu part_2;
    XpDecoderSeperatedPart_3_Xnn part_3;
};

// all in xnn
// 0.5B Can not work right now.
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

        norm = RMSNorm(config.hidden_size, (float)config.rms_norm_eps, names.post_norm_name);

        lm_head = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) { x = block({x})[0]; }

        // TODO how to handle norm and lm_head on NPU side.
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
            model_0_5 = QWenModel_0_5(config, names, names.blk_name);
        } else {
            model = QWenModel(config, names, names.blk_name);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        if (tie_embedding_words) {
            auto outputs = model_0_5({x})[0];
            return {outputs};
        } else {
            auto outputs = model({x})[0];
            return {outputs};
        }
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    QWenModel_0_5 model_0_5;
    QWenModel model;
};
