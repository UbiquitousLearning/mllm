//
// Created by Guo Xiaoqiang on 2024/8/12.
//
#ifndef MODELING_PHI3V_HPP
#define MODELING_PHI3V_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_phi3v.hpp"
#include "models/clip/modeling_clip.hpp"
#include "models/phi3/modeling_phi3.hpp"
// #include "models/vit/configuration_vit.hpp"
#include <string>

using namespace mllm;

class VisionEmbdReplace final : public Layer {
public:
    VisionEmbdReplace() = default;
    explicit VisionEmbdReplace(std::string name) {
        init(std::move(name), OpType::REPLACE);
    }
    Tensor operator()(Tensor text, Tensor vision, Tensor where_indices) {
        auto ts = run({text, vision, where_indices}, 1);
        return ts[0];
    }
};
class Phi3ImageEmbedding final : public Module {
public:
    CLipVisionModel img_processor;
    Embedding embed_tokens;
    VisionEmbdReplace embd_replace;
    int image_dim_out;
    Layer img_projector_linear1;
    Layer img_projector_relu;
    Layer img_projector_linear2;
    string project_cls;
    

    Phi3ImageEmbedding(int vocab_size, int hidden_dim, int head_size, int ffn,int img_dim, string &projection_cls, const Phi3VNameConfig &nameconfig, const string &base_name, const string &embd_name) :
        embed_tokens(vocab_size, hidden_dim, embd_name) {
        image_dim_out = img_dim;
        project_cls = projection_cls;
        if (project_cls == "Linear") {
            img_projector_linear1 = Linear(hidden_dim, hidden_dim, false, nameconfig._vision_model_prefix+nameconfig._projection+".0");
        } else if (project_cls == "MLP") {
            img_projector_linear1 = Linear(hidden_dim, hidden_dim, false, nameconfig._projection);
            img_projector_relu = GELU(nameconfig._projection);
            img_projector_linear2 = Linear(hidden_dim, hidden_dim, false, nameconfig._projection);
        } else {
            throw std::runtime_error("Unsupported projection_cls");
        }
        // img_processor = CLipVisionModel(img_dim, vision_head_size, vision_ffn_hidden, "QuickGELU", patch, img_hw, vision_block_num,
        //                                 nameconfig, nameconfig.vison_model_name);

        img_processor = CLipVisionModel(hidden_dim,head_size, ffn, "QuickGELU", 14, 336, 12, nameconfig, nameconfig.vison_model_name);
        embd_replace = VisionEmbdReplace("embd_replace");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        //  return {Tokenizer::tokens2Input(tokens_ids, std::move(text_name)), img_tensor,imgpos2Tensor(img_pos)};
    
        // img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
        // img_feature = img_processor_output.hidden_states[LAYER_IDX]

        // if TYPE_FEATURE == "patch":
        //     patch_feature = img_feature[:, 1:]
        //     return patch_feature
        bool have_img = inputs.size() > 1;
        auto text_features = embed_tokens({inputs[0]});
        if (have_img) {
            auto image_features = img_processor({inputs[1]})[0];
            // TODO check global_image_features
            // TODO check shape and all magic numbers
            // img projection
            auto image_features_proj = img_projector_linear1(image_features);
            if(project_cls == "MLP") {
                image_features_proj = img_projector_relu(image_features_proj);
                image_features_proj = img_projector_linear2(image_features_proj);
            }
            // TODO check dimension
            for (int i = 0; i < inputs[2].sequence(); i++) {
                auto where_idx = inputs[0].where(32044 * (i + 1), SEQUENCE);
                // TODO 现在这个实现只支持只有一图的情况
                text_features = embd_replace(text_features, image_features_proj, where_idx);
            }
        }

        return {text_features};
    }
};

class Phi3VModel final : public Module {
    Phi3ImageEmbedding vision_embed_tokens;
    vector<Phi3Block> blocks;
    RMSNorm norm;
    Layer lm_head;
    string embed_layer;

public:
    explicit Phi3VModel(const Phi3VConfig &config) :
        Phi3VModel(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit, config.img_dim, config.projection_cls, config.vision_model_config,
                   config.names_config, config.names_config.blk_name) {
    }
    Phi3VModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, int img_dim, string projection_cls, const Phi3VNameConfig &visionconfig,
               const Phi3NameConfig &names, const string &base_name) :
        norm(hidden_dim, 1e-6, names.post_norm_name), vision_embed_tokens(vocab_size, hidden_dim, head_size,ffn_hidden, img_dim, projection_cls, visionconfig, base_name, names.token_embd_name) {
        blocks = List<Phi3Block>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = vision_embed_tokens(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
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