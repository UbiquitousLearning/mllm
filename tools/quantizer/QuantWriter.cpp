#include "QuantWriter.hpp"
#include "Types.hpp"
#include "backends/cpu/compute/GemmKleidiai.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

namespace mllm {

const std::vector<std::string> fp32_layers = {
    "norm",
    "rope",
    "bias",
    "rotary_emb",
    // "embed_tokens",
    "_GN",
    "class_embedding",
    "embeddings",
    "logit_scale",
    "modality_preprocessors",
    "modality_heads",
    "modality_postprocessors",
    "pre_transformer_layer",
    "pos_embed.inv_freq",
    "ln_q",
    "patch_embed.proj",
    // "mlp.gate.",
    // "lm_head.weight",  // T
    // "query_key_value", // T
    // "word_embeddings", // T
};
const std::vector<std::string> q40_layers = {
    "embed_tokens",
    "word_embeddings",
};
const std::vector<std::string> q6_layers = {
    // "w2", "wv", "dense_h_to_4h", "v_proj", "down_proj",
};
const std::vector<std::string> q23_layers = {
    // ".experts.",
};
const std::vector<std::string> q23_to_q4_0_4x4_layers = {
    "w2",
    "wv",
    "dense_h_to_4h",
    "v_proj",
    "down_proj",
    "down",
};

const std::vector<std::string> q4_0_kai_to_q4_0_4x4_layers = {
    // 设置为KAI_Q4_0但不用 KAI_Q4_0
    "in_proj",
    "w12",
    "model.output",
    "merger.mlp",
    // for ling-lite-moe
    // "query_key_value",
    // "dense",
};

bool find_in_layers(const std::string &name, const std::vector<std::string> &layer_names) {
    if ("vision_embed_tokens" == name) return true;
    for (const auto &layer : layer_names) {
        if (name.find(layer) != std::string::npos) {
            return true;
        }
    }
    return false;
}

QuantWriter::QuantWriter(std::string output_path, std::string input_path) :
    ParamWriter(std::move(output_path)), output_path_(this->path_) {
    param_loader_ = new mllm::ParamLoader(std::move(input_path));
    if (!param_loader_->isAvailible()) {
        __exit(-1);
    }
}

QuantWriter::~QuantWriter() {
    delete param_loader_;
};

int QuantWriter::readParams() {
    original_param_names_ = param_loader_->getParamNames(); // 保存原始参数名
    param_names_ = original_param_names_;                   // 复制一份用于可能的操作

    // 检查 lm_head.weight 是否存在，如果不存在且 model.embed_tokens.weight 存在，则添加它
    bool lm_head_exists = false;
    bool embed_tokens_exists = false;
    for (const auto &name : original_param_names_) {
        if (name == "lm_head.weight") {
            lm_head_exists = true;
        }
        if (name == "model.embed_tokens.weight") {
            embed_tokens_exists = true;
        }
    }

    if (!lm_head_exists && embed_tokens_exists) {
        std::cout << "INFO: lm_head.weight not found, will be created by copying model.embed_tokens.weight" << std::endl;
        param_names_.push_back("lm_head.weight");
    }

    paddingIndex(param_names_);
    return param_names_.size();
}

std::vector<float> QuantWriter::load_full_fp32_param(const std::string &name) {
    if (param_loader_->getDataType(name) != MLLM_TYPE_F32) {
        return {};
    }
    auto [data_ptr, size] = param_loader_->load(name);
    if (data_ptr == nullptr || size == 0) {
        return {};
    }
    std::vector<float> param_data(size / sizeof(float));
    memcpy(param_data.data(), data_ptr, size);
    delete[] data_ptr;
    return param_data;
}

DataType QuantWriter::getQuantizationTypeFor(const std::string &name, DataType target_type, const std::string &other_flag) {
    /*
    if (name.find("down_proj") != std::string::npos && name.find("visual.blocks") != std::string::npos
        && name.find("bias") == std::string::npos) {
        return MLLM_TYPE_F32;
    }
    if (name.find("qkv") != std::string::npos && name.find("bias") == std::string::npos) {
        return MLLM_TYPE_Q4_K;
    }
    */
    if (find_in_layers(name, q40_layers)) {
        return MLLM_TYPE_Q4_0;
    }
    if (find_in_layers(name, fp32_layers)) {
        return MLLM_TYPE_F32;
    }
    if (find_in_layers(name, q23_layers) && (name.find("down") == std::string::npos)) {
        return MLLM_TYPE_Q2_K;
    }
    if (target_type == MLLM_TYPE_KLEIDIAI_Q4_0) {
        if (find_in_layers(name, q4_0_kai_to_q4_0_4x4_layers)) {
            return MLLM_TYPE_Q4_0_4_4; // MLLM_TYPE_Q4_0; // 这些层回退到Q4_0
        }
        if (other_flag == "eager" && name.find("v_proj") != std::string::npos) {
            return MLLM_TYPE_Q4_0; // eager模式下 v_proj 回退到Q4_0
        }
        return MLLM_TYPE_KLEIDIAI_Q4_0;
    }
    if (target_type >= MLLM_TYPE_Q2_K && target_type <= MLLM_TYPE_Q8_K) {
        if (find_in_layers(name, q6_layers)) {
            return MLLM_TYPE_Q6_K;
        }
    }
    return target_type;
}

void QuantWriter::quantize(DataType target_quant_type, const std::string &other_flag) {
    FILE *fp_in = param_loader_->getInputStream();
    if (!fp_in) {
        std::cout << "Failed to get input file stream from ParamLoader." << std::endl;
        __exit(-1);
    }

    const int CHUNK_SIZE_FLOATS = 1024 * 1024; // 每次处理4MB
    std::vector<float> read_buffer(CHUNK_SIZE_FLOATS);

    int tmp_hidden_dim = -1;
    int vit_tmp_hidden_dim = -1;
    int qw3_hidden_dim = 2048;

    // 预扫描以找到隐藏维度
    std::cout << "Pre-scanning to find hidden dimensions..." << std::endl;
    for (const auto &name : original_param_names_) {
        if (tmp_hidden_dim == -1 && (name.find("model") != std::string::npos && name.find("norm") != std::string::npos && name.find("k") == std::string::npos && name.find("q") == std::string::npos)) {
            ParamMetadata meta = param_loader_->getParamMetadata(name);
            tmp_hidden_dim = meta.size / sizeof(float);
            std::cout << "  - Found hidden dimension (tmp_hidden_dim): " << tmp_hidden_dim << " from layer '" << name << "'" << std::endl;
        }
        if (vit_tmp_hidden_dim == -1 && (name.find("visual") != std::string::npos && name.find("norm") != std::string::npos)) {
            ParamMetadata meta = param_loader_->getParamMetadata(name);
            vit_tmp_hidden_dim = meta.size / sizeof(float);
            std::cout << "  - Found ViT hidden dimension (vit_tmp_hidden_dim): " << vit_tmp_hidden_dim << " from layer '" << name << "'" << std::endl;
        }
        if (tmp_hidden_dim != -1 && vit_tmp_hidden_dim != -1) {
            break;
        }
    }

    for (const auto &name : param_names_) {
        bool is_copied_lm_head = (name == "lm_head.weight" && std::find(original_param_names_.begin(), original_param_names_.end(), name) == original_param_names_.end());
        DataType final_quant_type = getQuantizationTypeFor(name, target_quant_type, other_flag);

        std::cout << "Processing param " << name << " -> " << DataTypeName(final_quant_type) << " ... ";
        if (is_copied_lm_head) {
            std::cout << "(copied from model.embed_tokens.weight) ";
        }
        fflush(stdout);

        beginWriteParam(name, final_quant_type);

        std::vector<float> full_param_data;
        uint64_t num_floats;

        if (is_copied_lm_head) {
            full_param_data = load_full_fp32_param("model.embed_tokens.weight");
            if (full_param_data.empty()) {
                std::cerr << "FAIL! Failed to load model.embed_tokens.weight for copying." << std::endl;
                __exit(-1);
            }
            num_floats = full_param_data.size();
        } else {
            ParamMetadata meta = param_loader_->getParamMetadata(name);
            num_floats = meta.size / sizeof(float);
            fseek(fp_in, meta.offset, SEEK_SET);
            if (final_quant_type == MLLM_TYPE_KLEIDIAI_Q4_0 || final_quant_type == MLLM_TYPE_Q4_0_4_4) {
                full_param_data.resize(num_floats);
                fread(full_param_data.data(), sizeof(float), num_floats, fp_in);
            }
        }

        if (!full_param_data.empty()) {
            void *quant_ptr = nullptr;
            uint64_t quant_size = 0;

            if (final_quant_type == MLLM_TYPE_KLEIDIAI_Q4_0) {
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
                int H = find_in_layers(name, {"visual"}) ? vit_tmp_hidden_dim : tmp_hidden_dim;
                if (find_in_layers(name, {"self_attn.o_proj.weight"}) && other_flag == "qw3") {
                    H = qw3_hidden_dim;
                    std::cout << "(QWen3 self_attn.o_proj.weight detected, using hidden dim: " << H << ") ";
                }
                if (H <= 0) {
                    std::cout << "FAIL! Hidden dimension not found for " << name << std::endl;
                    __exit(-1);
                }

                // ==================【代码修正】==================
                // 恢复您指出的、用于判断 N 和 K 的关键逻辑
                int N, K;
                if (find_in_layers(name, {"w2", "down_proj", "down", "fc2"})) {
                    N = H;
                    if (num_floats % N != 0) {
                        std::cerr << "FAIL! num_floats " << num_floats << " not divisible by N for " << name << std::endl;
                        __exit(-1);
                    }
                    K = num_floats / N;
                } else {
                    K = H;
                    if (num_floats % K != 0) {
                        std::cerr << "FAIL! num_floats  " << num_floats << " not divisible by K for " << name << std::endl;
                        __exit(-1);
                    }
                    N = num_floats / K;
                }
                // ===============================================

                std::string bias_name = name;
                bias_name.replace(bias_name.find("weight"), 6, "bias");
                std::vector<float> bias_data = load_full_fp32_param(bias_name);
                std::vector<float> transposed_weight_data(num_floats);
                for (int n = 0; n < N; ++n)
                    for (int k = 0; k < K; ++k) transposed_weight_data[k * N + n] = full_param_data[n * K + k];
                auto block_t = alloc_kleidiai_quant_block(final_quant_type, N, K);
                quant_ptr = block_t.first;
                quant_size = block_t.second;
                // std::cout << "N: " << N << ", K: " << K << ", quant_size: " << quant_size << "  ";
#ifndef KAI_FP16_CAL
                mllm_kleidai_pack_b_and_bias_qsi4((uint8_t *)quant_ptr, transposed_weight_data.data(), bias_data.empty() ? nullptr : bias_data.data(), N, K);
#else
                mllm_kleidai_pack_b_and_bias_qsi4_to_fp16((uint8_t *)quant_ptr, transposed_weight_data.data(), bias_data.empty() ? nullptr : bias_data.data(), N, K);
#endif
#else
                std::cerr << "KLEIDIAI_Q4_0 is only supported on ARM architecture." << std::endl;
                __exit(-1);
#endif
            } else if (final_quant_type == MLLM_TYPE_Q4_0_4_4) {
                bool is_visual = find_in_layers(name, {"visual"});
                int H = is_visual ? vit_tmp_hidden_dim : tmp_hidden_dim;
                if (H <= 0) {
                    std::cout << "FAIL! Hidden dimension not found for " << name << std::endl;
                    __exit(-1);
                }
                int K = H;
                if ((is_visual && find_in_layers(name, {"fc2", "down_proj", "down"}))
                    || (!is_visual && find_in_layers(name, {"w2", "down_proj", "down"}))) {
                    if (num_floats % H != 0) {
                        std::cerr << "FAIL! num_floats not divisible by H for " << name << std::endl;
                        __exit(-1);
                    }
                    K = num_floats / H;
                }
                auto block_t = alloc_quant_block(num_floats, final_quant_type);
                quant_ptr = block_t.first;
                quant_size = block_t.second;
                quantize_row_q4_0_4x4(full_param_data.data(), quant_ptr, num_floats, K);
            } else {
                auto block_t = alloc_quant_block(num_floats, final_quant_type);
                quant_ptr = block_t.first;
                quant_size = block_t.second;
                switch (final_quant_type) {
                case MLLM_TYPE_F32: break;
                case MLLM_TYPE_Q4_0: quantize_row_q4_0(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q8_0: quantize_row_q8_0(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q2_K: quantize_row_q2_K(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q3_K: quantize_row_q3_K(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q4_K: quantize_row_q4_K(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q6_K: quantize_row_q6_K(full_param_data.data(), quant_ptr, num_floats); break;
                case MLLM_TYPE_Q8_K: quantize_row_q8_K(full_param_data.data(), quant_ptr, num_floats); break;
                default:
                    std::cerr << "Unsupported quantization type for full-tensor processing: " << DataTypeName(final_quant_type) << std::endl;
                    delete[] (char *)quant_ptr;
                    __exit(-1);
                }
            }
            if (final_quant_type == MLLM_TYPE_F32) {
                writeChunk(full_param_data.data(), num_floats * sizeof(float));
                if (quant_ptr) delete[] (char *)quant_ptr;
            } else {
                writeChunk(quant_ptr, quant_size);
                delete[] (char *)quant_ptr;
            }
        } else {
            uint64_t floats_processed = 0;
            while (floats_processed < num_floats) {
                uint64_t floats_to_read = std::min((uint64_t)CHUNK_SIZE_FLOATS, num_floats - floats_processed);
                if (floats_to_read == 0) break;
                fread(read_buffer.data(), sizeof(float), floats_to_read, fp_in);
                auto block_t = alloc_quant_block(floats_to_read, final_quant_type);
                void *quant_ptr = block_t.first;
                switch (final_quant_type) {
                case MLLM_TYPE_F32:
                    writeChunk(read_buffer.data(), floats_to_read * sizeof(float));
                    delete[] (char *)quant_ptr;
                    quant_ptr = nullptr;
                    break;
                case MLLM_TYPE_Q4_0: quantize_row_q4_0(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q8_0: quantize_row_q8_0(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q2_K: quantize_row_q2_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q3_K: quantize_row_q3_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q4_K: quantize_row_q4_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q6_K: quantize_row_q6_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q8_K: quantize_row_q8_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                default:
                    std::cerr << "Unsupported quantization type in streaming loop: " << DataTypeName(final_quant_type) << std::endl;
                    delete[] (char *)quant_ptr;
                    __exit(-1);
                }
                if (quant_ptr) {
                    writeChunk(quant_ptr, block_t.second);
                    delete[] (char *)quant_ptr;
                }
                floats_processed += floats_to_read;
            }
        }
        endWriteParam();
        std::cout << "Done." << std::endl;
    }
    writeIndex();
}
} // namespace mllm