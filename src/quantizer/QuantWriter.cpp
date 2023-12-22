#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/quantize/QuantizeQ4.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"
#include <string>
#include "QuantWriter.hpp"
#include "backends/cpu/quantize/QuantizeQ6.hpp"
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

vector<string> fp32_layers = {"norm", "rope", "bias", "vision_embed_tokens", "tok_embeddings"};
vector<string> q6_layers = {"w2", "wv", "dense_h_to_4h"};

bool find_names(const string &name, const vector<string> &layer_names) {
    for (const auto &layer : layer_names) {
        if (name.find(layer) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void QuantWriter::quantParams(DataType dataType) {
    quant_type_ = dataType;
    for (const auto &name : param_names_) {
        //        int force_quant_type = -1;
        auto *param = getParam(name);
        if (param == nullptr) {
            __exit(-1);
        }
        auto size = param_loader_->offsets_[name].second / sizeof(float);
        void *quant_ptr = nullptr;
        std::pair<void *, uint64_t> block_t;
        if(find_names(name, fp32_layers)) {
            std::cout << "Quantize param " << name << " to " << DataTypeName(MLLM_TYPE_F32) << "\t";
            const auto s = param_loader_->offsets_[name].second / sizeof(float);
            const auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
            writeParam(name, MLLM_TYPE_F32, param, tsize);
            std::cout << "  size:" << tsize << std::endl;
        }else if (find_names(name, q6_layers)) {
            switch (dataType) {
            case MLLM_TYPE_F32:
                std::cout << "No need to quantize FP32 params\n";
                __exit(-1);
                break;
            case MLLM_TYPE_Q4_0:
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
            }
            if (quant_ptr != nullptr) {
                if ((dataType == MLLM_TYPE_Q4_0) |(dataType ==MLLM_TYPE_Q4_K)|dataType ==MLLM_TYPE_Q6_K) {
                    writeParam(name, MLLM_TYPE_Q6_K, quant_ptr, size);
                    std::cout << "  size:" << size <<" type:"<< DataTypeName(MLLM_TYPE_Q6_K)<< std::endl;
                } else {
                    writeParam(name, quant_type_, quant_ptr, size);
                    std::cout << "  size:" << size <<" type:"<< DataTypeName(quant_type_)<< std::endl;
                }
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
void QuantWriter::writeParam(string name, DataType type, void *data, uint64_t size) {
#ifdef TEST
    data_[name] = (char *)data;
#endif
    ParamWriter::writeParam(name, type, data, size);
}

} // namespace mllm