//
// Created by Rongjie Yi on 24-6-20.
//

#ifndef MODELING_DEEPSEEK_HPP
#define MODELING_DEEPSEEK_HPP

#include "configuration_deepseek.hpp"

using namespace mllm;

class DeepseekMultiHeadLatentAttention final : public Module {
    Layer q_proj;
    Layer kv_a_proj_with_mqa;
    Layer kv_a_layernorm;
    Layer kv_b_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_rope;
    Layer k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    Layer o_proj;
    int num_heads{};
    int q_head_dim{};
    int v_head_dim{};
    int qk_nope_head_dim{};
    int qk_rope_head_dim{};
    int kv_lora_rank{};
    float softmax_scale{};
public:
    DeepseekMultiHeadLatentAttention() = default;
    DeepseekMultiHeadLatentAttention(const DeepseekConfig config, const DeepseekNameConfig &names, const string &base_name) {
        num_heads = config.num_heads;
        qk_nope_head_dim =config.qk_nope_head_dim;
        qk_rope_head_dim =config.qk_rope_head_dim;
        kv_lora_rank = config.kv_lora_rank;
        v_head_dim = config.v_head_dim;
        q_head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim;
        q_proj = Linear(
            config.hidden_size, 
            num_heads * q_head_dim, 
            false, 
            base_name + names._q_proj_name);
        kv_a_proj_with_mqa = Linear(
            config.hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            false,
            base_name + names._kv_a_proj_with_mqa_name
        );
        kv_a_layernorm = RMSNorm(kv_lora_rank, config.rms_norm_eps, base_name + names._kv_a_layernorm_name);
        kv_b_proj = Linear(
            kv_lora_rank,
            num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim),
            false,
            base_name + names._kv_b_proj_name
        );
        o_proj = Linear(
            num_heads * v_head_dim,
            config.hidden_size,
            false, 
            base_name + names._o_proj_name
        );        
        q_rope = RoPE(RoPEType::MLAROPE, base_name + "q_rope");
        k_rope = RoPE(RoPEType::MLAROPE, base_name + "k_rope");        
        if (config.cache_limit > 0) {
            k_cache = KVCache(num_heads/num_heads, config.cache_limit, base_name + "k_cache");
            v_cache = KVCache(num_heads/num_heads, config.cache_limit, base_name + "v_cache");
        }
        softmax = Softmax(DIMENSION, config.do_mask, base_name + "softmax");
        softmax_scale = 1/std::sqrt(q_head_dim);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        auto hidden_states = inputs[0];
    
        auto q = q_proj(hidden_states);
        auto qs = Tensor::split(q, {qk_nope_head_dim, qk_rope_head_dim}, D_HD, num_heads);
        q = Tensor::cat({qs[0], q_rope(qs[1])}, DIMENSION);

        Tensor compressed_kv = kv_a_proj_with_mqa(hidden_states);        
        auto kvs = Tensor::split(compressed_kv, 
                        {kv_lora_rank, qk_rope_head_dim}, DIMENSION);
        auto k_pe = k_rope(kvs[1]);
        auto kv = kv_b_proj(kv_a_layernorm(kvs[0]));//.view(-1, head_size_, -1, qk_nope_head_dim_ + v_head_dim_);
        kvs = Tensor::split(kv, {qk_nope_head_dim, v_head_dim}, D_HD, num_heads);
        auto v = kvs[1];
        auto k = Tensor::cat({kvs[0], k_pe}, DIMENSION);  
        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk * softmax_scale;
        qk = softmax(qk, k_cache.getCacheSeqLen());
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, v_head_dim * num_heads);
        o = o_proj(o);
        return {o};        
    }
};

class DeepseekMLP final : public Module {
private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer gelu;
public:
    DeepseekMLP() = default;
    DeepseekMLP(const DeepseekConfig &config, const DeepseekNameConfig &names, const std::string &base_name) {
        int hidden_size = config.hidden_size;
        int intermediate_size = config.intermediate_size;
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        gelu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = gelu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }
};

class DeepseekDecoder final : public Module {
private:
    DeepseekMultiHeadLatentAttention self_atten;
    DeepseekMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
public:
    DeepseekDecoder() = default;
    DeepseekDecoder(const DeepseekConfig &config, const DeepseekNameConfig &names, const string &base_name) {
        self_atten = DeepseekMultiHeadLatentAttention(config, names, base_name + names._attn_base_name);
        mlp = DeepseekMLP(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
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
};

class DeepseekModel final : public Module {
private:
    std::vector<DeepseekDecoder> blocks;
    Layer norm;
public:
    DeepseekModel() = default;
    DeepseekModel(const DeepseekConfig &config, const DeepseekNameConfig &names, const string &base_name) {
        blocks = List<DeepseekDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, names.post_norm_name);
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

class DeepseekForCausalLM final : public Module {
private:
    int hidden_size;
    Layer embedding;
    Parameter lm_head;
    DeepseekModel model;
public:
    DeepseekForCausalLM(DeepseekConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = DeepseekModel(config, names, names.blk_name);

        // lm_head and tok_embedding is tied together.
        // They share same parameters. Use a Transpose to do the lm_head instead.
        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.lm_head_name + ".weight");
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        auto outputs = model({x})[0];
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {outputs};
    }
};


#endif // MODELING_DEEPSEEK_HPP
