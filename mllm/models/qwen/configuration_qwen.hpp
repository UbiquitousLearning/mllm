/**
 * @file configuration_gemma.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief configuration file of qwen llm.
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIG_QWEN_HPP
#define CONFIG_QWEN_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace mllm;

class QWenNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief QWen2 following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init(RoPEType type = RoPEType::HFHUBROPE) {
        switch (type) {
        case RoPEType::HFHUBROPE: {
            blk_name = "model.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "o_proj";
            _gate_proj_name = "gate_proj";
            _up_proj_name = "up_proj";
            _down_proj_name = "down_proj";
            _attn_norm_name = "input_layernorm";
            _ffn_norm_name = "post_attention_layernorm";
            token_embd_name = "model.embed_tokens";
            post_norm_name = "model.norm";
            lm_head_name = "lm_head";
            break;
        }
        case RoPEType::LLAMAROPE: /*the qwen2 is same to llama*/ {
            blk_name = "layers.";
            _attn_base_name = "attention.";
            _ffn_base_name = "feed_forward.";
            _q_proj_name = "wq";
            _k_proj_name = "wk";
            _v_proj_name = "wv";
            _o_proj_name = "wo";
            _gate_proj_name = "w1";
            _up_proj_name = "w3";
            _down_proj_name = "w2";
            _attn_norm_name = "attention_norm";
            _ffn_norm_name = "ffn_norm";
            token_embd_name = "tok_embeddings";
            post_norm_name = "norm";
            lm_head_name = "output";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported gemma RoPE type");
        }
        }
    }

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct QWenConfig : public TransformerConfig {
    explicit QWenConfig(int token_limit, string billions = "0.5B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "0.5b") {
            attention_dropout = 0.0;
            bos_token_id = 151643;
            eos_token_id = 151645;
            std::string hidden_act = "silu";
            hidden_size = 1024;
            initializer_range = 0.02;
            intermediate_size = 2816;
            max_position_embeddings = 32768;
            max_window_layers = 21;
            model_type = "qwen2";
            num_attention_heads = 16;
            num_hidden_layers = 24;
            num_key_value_heads = 16;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = true;
        } else if ((billionsType == "0.5b-lm")
                   || (billionsType == "0.5b-rotated")) {
            attention_dropout = 0.0;
            bos_token_id = 151643;
            eos_token_id = 151645;
            std::string hidden_act = "silu";
            hidden_size = 1024;
            initializer_range = 0.02;
            intermediate_size = 2816;
            max_position_embeddings = 32768;
            max_window_layers = 21;
            model_type = "qwen2";
            num_attention_heads = 16;
            num_hidden_layers = 24;
            num_key_value_heads = 16;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else if (billionsType == "1.8b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 2048;
            intermediate_size = 5504;
            max_position_embeddings = 32768;
            num_attention_heads = 16;
            num_hidden_layers = 24;
            num_key_value_heads = 16;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else if (billionsType == "1.8b-rotated") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 2048;
            intermediate_size = 5504;
            max_position_embeddings = 32768;
            num_attention_heads = 16;
            num_hidden_layers = 24;
            num_key_value_heads = 16;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else if (billionsType == "1.5b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 1536;
            intermediate_size = 8960;
            max_position_embeddings = 32768;
            max_window_layers = 28;
            num_attention_heads = 12;
            num_hidden_layers = 28;
            num_key_value_heads = 2;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = true;
        } else if ((billionsType == "1.5b-rotated") || (billionsType == "1.5b-lm") || (billionsType == "1.5b-vl") || (billionsType == "1.5b-vl-rotated")) {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 1536;
            intermediate_size = 8960;
            max_position_embeddings = 32768;
            max_window_layers = 28;
            num_attention_heads = 12;
            num_hidden_layers = 28;
            num_key_value_heads = 2;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else if (billionsType == "3b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 2048;
            intermediate_size = 11008;
            max_position_embeddings = 32768;
            max_window_layers = 70;
            num_attention_heads = 16;
            num_hidden_layers = 36;
            num_key_value_heads = 2;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 151936;
            tie_embedding_words = true;
        } else if (billionsType == "ds-1.5b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 1536;
            intermediate_size = 8960;
            max_position_embeddings = 131072;
            max_window_layers = 28;
            num_attention_heads = 12;
            num_hidden_layers = 28;
            num_key_value_heads = 2;
            rms_norm_eps = 1e-6;
            rope_theta = 10000.f;
            sliding_window = 131072;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else if (billionsType == "7b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 3584;
            intermediate_size = 18944;
            max_position_embeddings = 32768;
            max_window_layers = 28;
            num_attention_heads = 28;
            num_hidden_layers = 28;
            num_key_value_heads = 4;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            sliding_window = 32768;
            vocab_size = 152064;
            tie_embedding_words = false;
        } else {
            throw std::runtime_error("QWenConfig Unsupported model size");
        }
        RoPE_type = type;
    };

    float attention_dropout = 0.0;
    int bos_token_id = 151643;
    int eos_token_id = 151643;
    std::string hidden_act = "silu";
    int hidden_size = 1024;
    float initializer_range = 0.02;
    int intermediate_size = 2816;
    int max_position_embeddings = 32768;
    int max_window_layers = 21;
    std::string model_type = "qwen2";
    int num_attention_heads = 16;
    int num_hidden_layers = 24;
    int num_key_value_heads = 16;
    double rms_norm_eps = 1e-6;
    float rope_theta = 1000000.0;
    int sliding_window = 32768;
    int vocab_size = 151936;
    bool tie_embedding_words = false;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    QWenNameConfig names_config;
};

struct QWenNPUConfig : virtual public QWenConfig {
    explicit QWenNPUConfig(int token_limit, string billions = "1.8B", RoPEType type = RoPEType::HFHUBROPE) :
        QWenConfig(token_limit, billions, type) {
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if ((billionsType == "0.5b-lm")
            || (billionsType == "0.5b-rotated")) {
            // Rotation folds the final RMSNorm into a cloned lm_head, so the
            // exported model no longer shares lm_head and token embeddings.
            shadow_layers = {};
            use_i32_bias = false;
            use_high_precision_silu = true;
        } else if (billionsType == "1.8b") {
            shadow_layers = {1, 2, 26};
        } else if (billionsType == "1.8b-rotated") {
            // Preserve the unclipped boundaries used by the published 1.8B
            // checkpoint and keep its W8A16 activation path.
            shadow_layers = {1, 2, 6};
            use_i32_bias = false;
            use_i16_attention_output = true;
            use_i16_mlp_activations = true;
            use_high_precision_silu = true;
            if (const char *override_layers =
                    std::getenv("MLLM_QWEN18_SHADOW_LAYERS")) {
                shadow_layers.clear();
                const std::string spec(override_layers);
                if (!spec.empty() && spec != "none") {
                    std::stringstream stream(spec);
                    std::string item;
                    while (std::getline(stream, item, ',')) {
                        if (item.empty()) {
                            throw std::invalid_argument(
                                "empty Qwen1.8B shadow layer override");
                        }
                        const int layer = std::stoi(item);
                        if (layer < 0 || layer >= num_hidden_layers) {
                            throw std::invalid_argument(
                                "Qwen1.8B shadow layer out of range");
                        }
                        shadow_layers.insert(layer);
                    }
                }
            }
            if (const char *i16_mlp_layers =
                    std::getenv("MLLM_QWEN18_I16_MLP_LAYERS")) {
                const std::string spec(i16_mlp_layers);
                if (!spec.empty() && spec != "none") {
                    std::stringstream stream(spec);
                    std::string item;
                    while (std::getline(stream, item, ',')) {
                        if (item.empty()) {
                            throw std::invalid_argument(
                                "empty Qwen1.8B I16 MLP layer override");
                        }
                        const int layer = std::stoi(item);
                        if (layer < 0 || layer >= num_hidden_layers) {
                            throw std::invalid_argument(
                                "Qwen1.8B I16 MLP layer out of range");
                        }
                        i16_mlp_layers_override.insert(layer);
                    }
                }
            }
        } else if (billionsType == "1.5b") { // qwen2.5 1.5B
            shadow_layers = {1, 2, 4, 5, 26};
            use_high_precision_silu = true;
        } else if (billionsType == "1.5b-vl") { // qwen-2-vl
            shadow_layers = {1, 26};
            use_high_precision_silu = false;
            tie_embedding_words = false;
        } else if (billionsType == "1.5b-rotated") { // qwen2.5 1.5B rotated model
            shadow_layers = {};
            use_i32_bias = false;
            use_high_precision_silu = true;
        } else if (billionsType == "1.5b-vl-rotated") { // qwen-2-vl rotated model
            shadow_layers = {};
            use_high_precision_silu = true;
            use_i32_bias = false;
            tie_embedding_words = false;
        } else {
            throw std::runtime_error("QWenNPUConfig Unsupported model size");
        }
    }

    std::set<int> shadow_layers;
    // use i32/fp32 bias for Linear in QNN, when using fp32 bias, bias will be added by DequantizeAdd
    bool use_i32_bias = true;
    // there are two types of QNNSiLU, a approximate int version and a (sigmoid * x) version
    // for qwen2.5, input of silu act has a large range, config here to use the (sigmoid * x)
    bool use_high_precision_silu = false;
    // Compatibility controls for calibration-sensitive Qwen1.5 exports.
    bool use_i16_attention_output = false;
    bool use_i16_mlp_activations = false;
    std::set<int> i16_mlp_layers_override;

    bool useI16MLPActivationsForLayer(int layer_idx) const {
        return use_i16_mlp_activations
            || i16_mlp_layers_override.find(layer_idx)
                != i16_mlp_layers_override.end();
    }

    // Sparse prefill is opt-in. Decoding remains dense.
    float prefill_attention_sparsity = 0.0F;
    bool prefill_attention_pattern_sparse = false;
    bool prefill_attention_hmx_selector = false;
    bool prefill_attention_keep_dense_edge_layers = true;
    std::vector<float> prefill_attention_layer_sparsities;
    std::vector<std::vector<float>> prefill_attention_head_retentions;

    std::vector<float> prefillAttentionHeadRetentionsForLayer(
        int layer_idx) const {
        if (prefill_attention_head_retentions.empty()) return {};
        if (prefill_attention_head_retentions.size()
            != static_cast<std::size_t>(num_hidden_layers)) {
            throw std::invalid_argument(
                "Qwen per-head retention layer count must match the model");
        }
        if (layer_idx < 0 || layer_idx >= num_hidden_layers) return {};
        const auto &retentions = prefill_attention_head_retentions[layer_idx];
        if (retentions.size()
            != static_cast<std::size_t>(num_attention_heads)) {
            throw std::invalid_argument(
                "Qwen per-head retention count must match the model");
        }
        for (float retention : retentions) {
            if (!(retention > 0.0F && retention <= 1.0F)) {
                throw std::invalid_argument(
                    "Qwen per-head retention must be in (0, 1]");
            }
        }
        return retentions;
    }

    float prefillAttentionSparsityForLayer(int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= num_hidden_layers) return 0.0F;
        if (!prefill_attention_layer_sparsities.empty()) {
            if (prefill_attention_layer_sparsities.size()
                != static_cast<std::size_t>(num_hidden_layers)) {
                throw std::invalid_argument(
                    "Qwen per-layer sparsity count must match the model");
            }
            const float sparsity =
                prefill_attention_layer_sparsities[layer_idx];
            if (!(sparsity >= 0.0F && sparsity < 1.0F)) {
                throw std::invalid_argument(
                    "Qwen per-layer sparsity must be in [0, 1)");
            }
            return sparsity;
        }
        if (!(prefill_attention_sparsity > 0.0F
              && prefill_attention_sparsity < 1.0F)) {
            return 0.0F;
        }
        if (!prefill_attention_keep_dense_edge_layers) {
            return prefill_attention_sparsity;
        }
        return layer_idx >= 2 && layer_idx < num_hidden_layers - 1
            ? prefill_attention_sparsity
            : 0.0F;
    }

    bool useSparsePrefillAttention(int layer_idx) const {
        const auto retentions =
            prefillAttentionHeadRetentionsForLayer(layer_idx);
        if (!retentions.empty()) {
            return std::any_of(
                retentions.begin(), retentions.end(),
                [](float retention) { return retention < 1.0F; });
        }
        return prefillAttentionSparsityForLayer(layer_idx) > 0.0F;
    }
};

#endif //! CONFIG_QWEN_HPP
