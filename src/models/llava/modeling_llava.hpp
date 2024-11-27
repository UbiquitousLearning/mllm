//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef MODELING_LLAVA_HPP
#define MODELING_LLAVA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llava.hpp"
#include "models/llama/modeling_llama.hpp"
#include "models/vit/modeling_vit.hpp"

using namespace mllm;

class LLaMABodyModel final : public Module {
    vector<LLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    LLaMABodyModel() = default;
    LLaMABodyModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
                   const LLaMANameConfig &names, const string &base_name) {
        blocks = List<LLaMABlock>(block_num, hidden_dim, head_size, head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

class LLaVAVisionEmbedding final : public Module {
    Layer patch_embedding;
    Parameter cls_token;
    Layer position_embedding;
    int range_len_{};

public:
    LLaVAVisionEmbedding() = default;
    LLaVAVisionEmbedding(int hidden_dim, int patch, int img_hw, const ViTNameConfig &names, const string &base_name) {
        patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        range_len_ = std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1;
        position_embedding = Embedding(range_len_, hidden_dim, base_name + names._position_embeddings_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose({{SEQUENCE, DIMENSION}, {HEAD, SEQUENCE}});
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        auto position_ids = Tensor::range(0, range_len_);
        embd = position_embedding(position_ids) + embd;
        return {embd};
    }
};

class LLaVAVisionModel final : public Module {
    LLaVAVisionEmbedding embedding;
    Layer pre_layrnorm;
    vector<ViTBlock> blocks;
    Layer linear_1;
    Layer gelu;
    Layer linear_2;
    int clip_len_{};

public:
    LLaVAVisionModel() = default;
    LLaVAVisionModel(int hidden_dim, int head_size, int ffn_hidden, int patch, int img_hw, int block_num,
                     const ViTNameConfig &names, const string &base_name) {
        embedding = LLaVAVisionEmbedding(hidden_dim, patch, img_hw, names, base_name + names._embd_name);
        pre_layrnorm = LayerNorm(hidden_dim, true, 1e-6, base_name + names._vision_pre_layrnorm_name);
        blocks = List<ViTBlock>(block_num, hidden_dim, head_size, ffn_hidden, "QuickGELU", names, base_name + names._layer_name);
        clip_len_ = std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1;
        linear_1 = Linear(hidden_dim, ffn_hidden, true, "multi_modal_projector.linear_1");
        gelu = GELU("multi_modal_projector.act");
        linear_2 = Linear(ffn_hidden, ffn_hidden, true, "multi_modal_projector.linear_2");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        x = pre_layrnorm(x);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {1, clip_len_}, {});
        x = linear_1(x);
        x = gelu(x);
        x = linear_2(x);
        return {x};
    }
};
class LLaVAModel final : public Module {
    Layer text_embedding;
    LLaVAVisionModel vision_tower;
    LLaMABodyModel llama_body;

public:
    explicit LLaVAModel(const LLaVAConfig &config) :
        LLaVAModel(config.vocab_size, config.hidden_dim, config.head_size, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                   config.names_config,
                   config.vision_hidden_dim, config.vision_head_size, config.vision_ffn_hidden, config.patch, config.img_hw, config.vision_block_num,
                   config.vit_names_config) {
    }
    LLaVAModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num,
               RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
               const LLaMANameConfig &names_config,
               int vision_hidden_dim, int vision_head_size, int vision_ffn_hidden, int patch, int img_hw, int vision_block_num,
               const ViTNameConfig &vit_names_config) {
        text_embedding = Embedding(vocab_size, hidden_dim, names_config.token_embd_name);
        llama_body = LLaMABodyModel(vocab_size, hidden_dim, head_size, ffn_hidden, block_num,
                                    RoPE_type, rope_theta, max_position_embeddings, cache_limit,
                                    names_config, names_config.blk_name);
        vision_tower = LLaVAVisionModel(vision_hidden_dim, vision_head_size, vision_ffn_hidden, patch, img_hw, vision_block_num,
                                        vit_names_config, vit_names_config.vison_model_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = text_embedding(inputs[0]);
        if (inputs[1].batch() > 0) {
            auto vision = vision_tower({inputs[1]})[0];
            auto where_idx = inputs[0].where(32000, SEQUENCE);
            embd = embd.index_put(vision, where_idx, true);
        }
        embd = llama_body({embd})[0];
        embd = embd.clip({}, {}, {-1}, {});
        return {embd};
    }
};

#endif // MODELING_LLAVA_HPP
