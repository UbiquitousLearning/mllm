#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "Types.hpp"
#include <string>
#include "QuantWriter.hpp"
#include "backends/cpu/compute/GEMM_AArch64.hpp"
namespace mllm {
QuantWriter::QuantWriter(std::string output_path, std::string input_path) :
    ParamWriter(output_path), output_path_(output_path) {
    param_loader_ = new mllm::ParamLoader(std::move(input_path));
    if (param_loader_ == nullptr) {
        __exit(-1);
    }
}
QuantWriter::~QuantWriter() {
#ifdef TEST
    for (auto &item : data_) {
        delete[] item.second;
    }
#endif
};
int QuantWriter::readParams() {
    param_names_ = param_loader_->getParamNames();
    paddingIndex(param_names_);
    return param_names_.size();
}
float *QuantWriter::getParam(std::string param_name) {
    auto type = param_loader_->data_type_[param_name];
    if (type != DataType::MLLM_TYPE_F32) {
        return nullptr;
    }
    auto [data, size] = param_loader_->load(param_name);
    return static_cast<float *>((void *)data);
}
bool find_names(const string &name, const vector<string> &layer_names) {
    if ("vision_embed_tokens" == name) { // for FUYU
        return true;
    }
    for (const auto &layer : layer_names) {
        if (name.find(layer) != std::string::npos) {
            return true;
        }
    }
    return false;
}

vector<string> fp32_layers = {
    "norm",
    "rope",
    "bias",
    "rotary_emb",
    "embed_tokens",
    "_GN",
    "class_embedding",
    // "vision_embed_tokens",
    "embeddings",
    "logit_scale",
    //"tok_embeddings",
    "modality_preprocessors",
    "modality_heads",
    "modality_postprocessors",
    "pre_transformer_layer",
    "pos_embed.inv_freq",
    "ln_q",
    "patch_embed.proj",
    "lm_head.weight",
    // MoE
    "mlp.gate.",
};
vector<string> q6_layers = {
    "w2",
    "wv",
    "dense_h_to_4h",
    "v_proj",
    "down_proj",
};
vector<string> q2k_layers = {
    // "mlp.experts",
};

vector<string> q3k_layers = {
    // "mlp.experts",
};

int tmp_hidden_dim = -1;
void QuantWriter::quantParams(DataType dataType) {
    quant_type_ = dataType;
    for (const auto &name : param_names_) {
        //        int force_quant_type = -1;
        auto *param = getParam(name);
        if (param == nullptr) {
            __exit(-1);
        }
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        if (find_names(name, {"input_layernorm"})) {
            tmp_hidden_dim = size;
        }
        void *quant_ptr = nullptr;
        std::pair<void *, uint64_t> block_t;
        if (find_names(name, q6_layers) && (dataType == MLLM_TYPE_Q6_K || dataType == MLLM_TYPE_Q4_K || dataType == MLLM_TYPE_Q2_K || dataType == MLLM_TYPE_Q3_K)) {
            if (tmp_hidden_dim > 0 && (size / tmp_hidden_dim) % 256 != 0) {
                std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q4_0) << "\t";
                block_t = alloc_quant_block(size, MLLM_TYPE_Q4_0);
                quant_ptr = block_t.first;
                quantize_row_q4_0(param, quant_ptr, size);
                size = block_t.second;
                if (quant_ptr != nullptr) {
                    writeParam(name, MLLM_TYPE_Q4_0, quant_ptr, size);
                    std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q4_0) << std::endl;
                }

                continue;
            }
        }
        if (find_names(name, fp32_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_F32) << "\t";
            const auto s = param_loader_->offsets_[name].second / sizeof(float);
            const auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
            writeParam(name, MLLM_TYPE_F32, param, tsize);
            std::cout << "  size:" << tsize << std::endl;

        } else if (find_names(name, q6_layers)) {
            switch (dataType) {
            case MLLM_TYPE_F32:
                std::cout << "No need to quantize FP32 params\n";
                __exit(-1);
                break;
            case MLLM_TYPE_Q4_0:
                std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q4_0(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q4_K:
            case MLLM_TYPE_Q6_K:
                std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q6_K) << "\t";
                block_t = alloc_quant_block(size, MLLM_TYPE_Q6_K);
                quant_ptr = block_t.first;
                quantize_row_q6_K(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q8_0:
                std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q8_0(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q8_K:
                std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q8_K(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q4_0_4_4:
                std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q4_0_4x4(param, quant_ptr, size);
                size = block_t.second;
                break;
            default:
                break;
            }
            if (quant_ptr != nullptr) {
                if ((dataType == MLLM_TYPE_Q4_K) | (dataType == MLLM_TYPE_Q6_K)) {
                    writeParam(name, MLLM_TYPE_Q6_K, quant_ptr, size);
                    std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q6_K) << std::endl;
                } else {
                    writeParam(name, quant_type_, quant_ptr, size);
                    std::cout << "  size:" << size << " type:" << DataTypeName(quant_type_) << std::endl;
                }
            }
        } else if (find_names(name, q3k_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q3_K) << "\t";
            block_t = alloc_quant_block(size, dataType);
            quant_ptr = block_t.first;
            quantize_row_q3_K(param, quant_ptr, size);
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, MLLM_TYPE_Q3_K, quant_ptr, size);
                std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q3_K) << std::endl;
            }
        } else if (find_names(name, q2k_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q2_K) << "\t";
            block_t = alloc_quant_block(size, dataType);
            quant_ptr = block_t.first;
            quantize_row_q2_K(param, quant_ptr, size);
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, MLLM_TYPE_Q2_K, quant_ptr, size);
                std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q2_K) << std::endl;
            }
        } else {
            std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
            switch (dataType) {
            case MLLM_TYPE_F32:
                std::cout << "No need to quantize FP32 params\n";
                __exit(-1);
                break;
            case MLLM_TYPE_Q4_0:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q4_0(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q8_0:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q8_0(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q4_K:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q4_K(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q6_K:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q6_K(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q8_K:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q8_K(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_Q4_0_4_4:
                block_t = alloc_quant_block(size, dataType);
                quant_ptr = block_t.first;
                quantize_row_q4_0_4x4(param, quant_ptr, size);
                size = block_t.second;
                break;
            case MLLM_TYPE_I8:
            case MLLM_TYPE_Q4_1:
            case MLLM_TYPE_Q8_1:
            case MLLM_TYPE_I16:
            case MLLM_TYPE_I32:
            case MLLM_TYPE_F16:
                NOT_IMPLEMENTED(dataType);
                break;
            case MLLM_TYPE_COUNT:
                UNREACHABLE()
                break;
            default:
                break;
            }
            if (quant_ptr != nullptr) {
                writeParam(name, quant_type_, quant_ptr, size);
                std::cout << "  size:" << size << std::endl;
            }
            // writeParam(name, quant_type_, quant_ptr, size);
#ifndef TEST
            delete[] (char *)quant_ptr;
#endif
        }
    }
    writeIndex();
}
vector<string> q4x4_2_q4_layers = {
    "wv",
    "v_proj",
    "qkv_proj",
    // dclm
    "in_proj",
    "w12",
    "model.output",
};
bool dclm_flag = false;
void QuantWriter::quantParams_q4_(DataType dataType) {
    bool do_quantParams_q4_vl = false;
    for (const auto &name : param_names_) {
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        if (find_names(name, {"visual"})) {
            do_quantParams_q4_vl = true;
        }
    }
    if (do_quantParams_q4_vl) {
        return quantParams_q4_vl(dataType);
    }

    for (const auto &name : param_names_) {
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        if (find_names(name, {"norm"})) {
            tmp_hidden_dim = size;
        }
        if (find_names(name, {"w12"})) {
            dclm_flag = true;
        }
    }
    quant_type_ = dataType;
    for (const auto &name : param_names_) {
        auto *param = getParam(name);
        if (param == nullptr) {
            __exit(-1);
        }
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        if (find_names(name, {"norm"})) {
            tmp_hidden_dim = size;
        }
        void *quant_ptr = nullptr;
        std::pair<void *, uint64_t> block_t;
        if (find_names(name, fp32_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_F32) << "\t";
            const auto s = param_loader_->offsets_[name].second / sizeof(float);
            const auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
            writeParam(name, MLLM_TYPE_F32, param, tsize);
            std::cout << "  size:" << tsize << std::endl;
        } else if (find_names(name, q4x4_2_q4_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q4_0) << "\t";
            block_t = alloc_quant_block(size, MLLM_TYPE_Q4_0);
            quant_ptr = block_t.first;
            quantize_row_q4_0(param, quant_ptr, size);
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, MLLM_TYPE_Q4_0, quant_ptr, size);
                std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q4_0) << std::endl;
            }
        } else {
            std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
            block_t = alloc_quant_block(size, dataType);
            quant_ptr = block_t.first;
            int tmp_hidden_dim_q4 = tmp_hidden_dim;
            if (find_names(name, {"w2", "down_proj"}) || (dclm_flag && find_names(name, {"w3"}))) {
                tmp_hidden_dim_q4 = (size / tmp_hidden_dim);
            }
            quantize_row_q4_0_4x4(param, quant_ptr, size, tmp_hidden_dim_q4);
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, quant_type_, quant_ptr, size);
                std::cout << "  size:" << size << std::endl;
            }
#ifndef TEST
            delete[] (char *)quant_ptr;
#endif
        }
    }
    writeIndex();
}

vector<string> vl_q4x4_2_q4_layers = {
    "wv",
    "v_proj",
    ".attn.qkv",
    "in_proj",
    "w12",
    "model.output",
    // "embed_tokens",
    "mlp.0",
    "mlp.2",

    // "visual",
    // "q_proj",
    // "model",
};
int vit_tmp_hidden_dim = -1;
void QuantWriter::quantParams_q4_vl(DataType dataType) {
    for (const auto &name : param_names_) {
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        if (find_names(name, {"input_layernorm"}) && find_names(name, {"model"})) {
            tmp_hidden_dim = size;
        }
        if (find_names(name, {"norm"}) && find_names(name, {"visual"})) {
            vit_tmp_hidden_dim = size;
        }
    }
    std::cout << "tmp_hidden_dim:" << tmp_hidden_dim << std::endl;
    std::cout << "vit_tmp_hidden_dim:" << vit_tmp_hidden_dim << std::endl;
    quant_type_ = dataType;
    for (const auto &name : param_names_) {
        auto *param = getParam(name);
        if (param == nullptr) {
            __exit(-1);
        }
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        // if (find_names(name, {"input_layernorm"})) {
        //     tmp_hidden_dim = size;
        // }
        void *quant_ptr = nullptr;
        std::pair<void *, uint64_t> block_t;
        if (find_names(name, fp32_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_F32) << "\t";
            const auto s = param_loader_->offsets_[name].second / sizeof(float);
            const auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
            writeParam(name, MLLM_TYPE_F32, param, tsize);
            std::cout << "  size:" << tsize << std::endl;
        } else if (find_names(name, vl_q4x4_2_q4_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_Q4_0) << "\t";
            block_t = alloc_quant_block(size, MLLM_TYPE_Q4_0);
            quant_ptr = block_t.first;
            quantize_row_q4_0(param, quant_ptr, size);
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, MLLM_TYPE_Q4_0, quant_ptr, size);
                std::cout << "  size:" << size << " type:" << DataTypeName(MLLM_TYPE_Q4_0) << std::endl;
            }
        } else {
            std::cout << "Quantize param " << name << " to " << DataTypeName(dataType) << "\t";
            block_t = alloc_quant_block(size, dataType);
            quant_ptr = block_t.first;
            if (find_names(name, {"visual"})) {
                int tmp_hidden_dim_q4 = vit_tmp_hidden_dim;
                if (find_names(name, {"fc2", "down_proj"})) {
                    tmp_hidden_dim_q4 = (size / vit_tmp_hidden_dim);
                }
                quantize_row_q4_0_4x4(param, quant_ptr, size, tmp_hidden_dim_q4);
            } else {
                int tmp_hidden_dim_q4 = tmp_hidden_dim;
                if (find_names(name, {"w2", "down_proj"})) {
                    tmp_hidden_dim_q4 = (size / tmp_hidden_dim);
                }
                quantize_row_q4_0_4x4(param, quant_ptr, size, tmp_hidden_dim_q4);
            }
            size = block_t.second;
            if (quant_ptr != nullptr) {
                writeParam(name, quant_type_, quant_ptr, size);
                std::cout << "  size:" << size << std::endl;
            }
#ifndef TEST
            delete[] (char *)quant_ptr;
#endif
        }
    }
    writeIndex();
}

void QuantWriter::writeParam(string name, DataType type, void *data, uint64_t size) {
#ifdef TEST
    data_[name] = (char *)data;
#endif
    ParamWriter::writeParam(name, type, data, size);
}

} // namespace mllm