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
    // "lm_head.weight",
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
    param_names_ = param_loader_->getParamNames();
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
    if (find_in_layers(name, q40_layers)) {
        return MLLM_TYPE_Q4_0;
    }
    if (find_in_layers(name, fp32_layers)) {
        return MLLM_TYPE_F32;
    }
    if (find_in_layers(name, q23_layers) && (name.find("down_proj") == std::string::npos)) {
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

    for (const auto &name : param_names_) {
        ParamMetadata meta = param_loader_->getParamMetadata(name);
        const uint64_t num_floats = meta.size / sizeof(float);

        for (const auto &name : param_names_) {
            if (tmp_hidden_dim != -1 && vit_tmp_hidden_dim != -1) {
                break; // 如果都找到了，就提前退出预扫描
            }
            /*
            if (tmp_hidden_dim == -1 && find_in_layers(name, {"model.layers.0.input_layernorm"})) {
                ParamMetadata meta = param_loader_->getParamMetadata(name);
                tmp_hidden_dim = meta.size / sizeof(float);
            }
            if (vit_tmp_hidden_dim == -1 && find_in_layers(name, {"visual.patch_embed.norm"})) {
                ParamMetadata meta = param_loader_->getParamMetadata(name);
                vit_tmp_hidden_dim = meta.size / sizeof(float);
            }
            */
            if (tmp_hidden_dim == -1 && (name.find("model") != std::string::npos && name.find("norm") != std::string::npos)) {
                ParamMetadata meta = param_loader_->getParamMetadata(name);
                tmp_hidden_dim = meta.size / sizeof(float);
                std::cout << "  - Found hidden dimension (tmp_hidden_dim): " << tmp_hidden_dim << " from layer '" << name << "'" << std::endl;
            }
            if (vit_tmp_hidden_dim == -1 && (name.find("visual") != std::string::npos && name.find("norm") != std::string::npos)) {
                ParamMetadata meta = param_loader_->getParamMetadata(name);
                vit_tmp_hidden_dim = meta.size / sizeof(float);
                std::cout << "  - Found ViT hidden dimension (vit_tmp_hidden_dim): " << vit_tmp_hidden_dim << " from layer '" << name << "'" << std::endl;
            }
        }

        DataType final_quant_type = getQuantizationTypeFor(name, target_quant_type, other_flag);

        std::cout << "Processing param " << name << " -> " << DataTypeName(final_quant_type) << " ... ";
        fflush(stdout);

        beginWriteParam(name, final_quant_type);
        fseek(fp_in, meta.offset, SEEK_SET);

        if (final_quant_type == MLLM_TYPE_F32) {
            std::vector<char> f32_buffer(meta.size);
            fread(f32_buffer.data(), 1, meta.size, fp_in);
            writeChunk(f32_buffer.data(), meta.size);
        }
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
        else if (final_quant_type == MLLM_TYPE_KLEIDIAI_Q4_0) {
            int H = find_in_layers(name, {"visual"}) ? vit_tmp_hidden_dim : tmp_hidden_dim;
            if (H <= 0) {
                std::cout << "FAIL! Hidden dimension not found for " << name << std::endl;
                __exit(-1);
            }

            int N, K;
            if (find_in_layers(name, {"w2", "down_proj", "fc2"})) {
                N = H;
                K = num_floats / N;
            } else {
                K = H;
                N = num_floats / K;
            }

            // 1. 加载完整的 bias (因为它很小)
            std::string bias_name = name;
            bias_name.replace(bias_name.find("weight"), 6, "bias");
            std::vector<float> bias_data = load_full_fp32_param(bias_name);

            // 2. 分配转置矩阵的内存
            std::vector<float> transposed_weight_data(num_floats);

            // 3. 流式读取并转置
            std::vector<float> row_buffer(K);
            for (int n = 0; n < N; ++n) {
                fread(row_buffer.data(), sizeof(float), K, fp_in);
                for (int k = 0; k < K; ++k) {
                    transposed_weight_data[k * N + n] = row_buffer[k];
                }
            }

            // 4. 分配量化空间并执行量化
            auto block_t = alloc_kleidiai_quant_block(final_quant_type, N, K);
            void *quant_ptr = block_t.first;

#ifndef KAI_FP16_CAL
            mllm_kleidai_pack_b_and_bias_qsi4(
                (uint8_t *)quant_ptr,
                transposed_weight_data.data(),
                bias_data.empty() ? nullptr : bias_data.data(),
                N, K);
#else
            mllm_kleidai_pack_b_and_bias_qsi4_to_fp16(
                (uint8_t *)quant_ptr,
                transposed_weight_data.data(),
                bias_data.empty() ? nullptr : bias_data.data(),
                N, K);
#endif

            // 5. 写入量化数据块
            writeChunk(quant_ptr, block_t.second);
            delete[] (char *)quant_ptr;

        }
#endif
        else if (final_quant_type == MLLM_TYPE_Q4_0_4_4) {
            // 1. 为 Q4_0_4_4 加载完整的 FP32 数据
            std::vector<float> param_data(num_floats);
            fread(param_data.data(), sizeof(float), num_floats, fp_in);

            // 2. 确定维度信息 (借鉴旧代码逻辑)
            // 'vl' 标志和 'visual' 层名的判断
            bool is_visual = find_in_layers(name, {"visual"});
            int H = is_visual ? vit_tmp_hidden_dim : tmp_hidden_dim;
            if (H <= 0) {
                std::cout << "FAIL! Hidden dimension not found for " << name << std::endl;
                __exit(-1);
            }
            // 根据层名决定 K 的值
            int K = H;
            if ((is_visual && find_in_layers(name, {"fc2", "down_proj"})) || (!is_visual && find_in_layers(name, {"w2", "down_proj"}))) {
                K = num_floats / H;
            }

            // 3. 分配量化空间并执行量化
            auto block_t = alloc_quant_block(num_floats, final_quant_type);
            void *quant_ptr = block_t.first;

            // 调用与旧版本中类似的函数
            quantize_row_q4_0_4x4(param_data.data(), quant_ptr, num_floats, K);

            // 4. 写入量化数据块
            writeChunk(quant_ptr, block_t.second);
            delete[] (char *)quant_ptr;
        } else {
            uint64_t floats_processed = 0;
            while (floats_processed < num_floats) {
                uint64_t floats_to_read = std::min((uint64_t)CHUNK_SIZE_FLOATS, num_floats - floats_processed);
                if (floats_to_read == 0) break;

                fread(read_buffer.data(), sizeof(float), floats_to_read, fp_in);

                auto block_t = alloc_quant_block(floats_to_read, final_quant_type);
                void *quant_ptr = block_t.first;

                switch (final_quant_type) {
                case MLLM_TYPE_Q4_0: quantize_row_q4_0(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q8_0: quantize_row_q8_0(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q2_K: quantize_row_q2_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q2_0: quantize_row_q2_0(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q3_K: quantize_row_q3_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q4_K: quantize_row_q4_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q6_K: quantize_row_q6_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                case MLLM_TYPE_Q8_K: quantize_row_q8_K(read_buffer.data(), quant_ptr, floats_to_read); break;
                default:
                    std::cerr << "Unsupported quantization type in streaming loop: " << DataTypeName(final_quant_type) << std::endl;
                    delete[] (char *)quant_ptr; // 避免内存泄漏
                    __exit(-1);
                }

                writeChunk(quant_ptr, block_t.second);
                delete[] (char *)quant_ptr;

                floats_processed += floats_to_read;
            }
        }

        endWriteParam();
        std::cout << "Done." << std::endl;
    }

    writeIndex();
}
} // namespace mllm