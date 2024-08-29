//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_MINICPM_HPP
#define MODELING_MINICPM_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_minicpm.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cmath>

using namespace mllm;
int sharedVariable = 0;

class MiniCPMMLP final : public Module {
public:
    MiniCPMMLP() = default;
    MiniCPMMLP(int hidden_size, int intermediate_size, const MiniCPMNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        // if(x.allocted()) {
        //         x.printData<float>();
        //     }
        x = silu(x);
        // if(x.allocted()) {
        //         x.printData<float>();
        //     }
        auto y = up_proj(inputs[0]); //ERROR
        x = x * y;
        // if(x.allocted()) {
        //         x.printData<float>();
        //     }
        x = down_proj(x);
        // if(x.allocted()) {
        //         x.printData<float>();
        //     }
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer silu;
};

class MiniCPMAttention final : public Module {
public:
    MiniCPMAttention() = default;
    MiniCPMAttention(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        head_dim = config.head_dim;
        num_heads = config.num_attention_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, false, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, false, base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        q_rope = RoPE(config.RoPE_type, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, base_name + "k_rope");
        k_cache = KVCache(num_heads / num_key_value_heads, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_heads / num_key_value_heads, config.cache_limit, base_name + "v_cache");
        mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);

        // [batch, heads, sequence, dims]
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // embedding
        query_states = q_rope(query_states);  //ERROR
        // if(query_states.allocted()) {
        //         query_states.printData<float>();
        //     }
        key_states = k_rope(key_states); //ERROR
        // if(key_states.allocted()) {
        //         query_states.printData<float>();
        //     }

        // kv cache
        key_states = k_cache(key_states);
        // if(key_states.allocted()) {
        //         query_states.printData<float>();
        //     }
        value_states = v_cache(value_states);
        // if(value_states.allocted()) {
        //         query_states.printData<float>();
        //     }

        // attention weight
        auto atten_weight = Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION)) / std::sqrt(head_dim);
        // if(atten_weight.allocted()) {
        //         atten_weight.printData<float>();
        //     }
        atten_weight = mask(atten_weight);
        // if(atten_weight.allocted()) {
        //         atten_weight.printData<float>();
        //     }
        atten_weight = softmax(atten_weight);
        // if(atten_weight.allocted()) {
        //         atten_weight.printData<float>();
        //     }

        // attention output
        auto atten_output = Tensor::mm(atten_weight, value_states);
        // if(atten_output.allocted()) {
        //         atten_weight.printData<float>();
        //     }
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        // if(atten_output.allocted()) {
        //         atten_weight.printData<float>();
        //     }
        atten_output = o_proj(atten_output);
        // if(atten_output.allocted()) {
        //         atten_weight.printData<float>();
        //     }
        return {atten_output};
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
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer mask;
    Layer softmax;
};

class MiniCPMDecoder final : public Module {
public:
    MiniCPMDecoder() = default;
    MiniCPMDecoder(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, 
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                       config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit, 
                                       true, false, names, base_name + names._attn_base_name);   
        mlp = MiniCPMMLP(config.hidden_size, config.intermediate_size, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        scale_depth = config.scale_depth;
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto residual = inputs[0];
        // if(residual.allocted()) {
        //         residual.printData<float>();
        //     }
        auto hidden_states = input_layernorm(inputs[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        hidden_states = residual + hidden_states * (scale_depth / static_cast<float>(std::sqrt(static_cast<long double>(num_hidden_layers))));
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        residual = hidden_states;
        hidden_states = post_attention_layernorm(hidden_states);
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        hidden_states = mlp({hidden_states})[0]; //ERROR
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        hidden_states = residual + hidden_states * (scale_depth / static_cast<float>(std::sqrt(static_cast<long double>(num_hidden_layers))));
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        return {hidden_states};
    }

private:
    MultiHeadAttention self_atten;
    MiniCPMMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    float scale_depth;
    int num_hidden_layers;
};

class MiniCPMModel final : public Module {
public:
    MiniCPMModel() = default;
    MiniCPMModel(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        blocks = List<MiniCPMDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }

    //receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        // if(hidden_states.allocted()) {
        //         hidden_states.printData<float>();
        //     }
        int cnt = 0;
        for (auto &block : blocks) {
            hidden_states = block({hidden_states})[0];
            // std::cout << "block " << cnt++ << std::endl;
            // if(hidden_states.allocted()) {
            //     hidden_states.printData<float>();
            // }
        }
        hidden_states = norm(hidden_states);
        return {hidden_states};
    }

private:
    std::vector<MiniCPMDecoder> blocks;
    Layer norm;
};

class MiniCPMForCausalLM final : public Module {
public:
    MiniCPMForCausalLM(MiniCPMConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = MiniCPMModel(config, names, names.blk_name);
        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // go through model
        auto outputs = model({x})[0];
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {outputs};
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    MiniCPMModel model;
};

#endif // MODELING_MINICPM_HPP