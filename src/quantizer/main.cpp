//
// Created by lx on 23-10-31.
//
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/quantize/QuantizeQ4.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"
#include <string>

#define NOT_IMPLEMENTED(x)                                                            \
    std::cout << "Quantize params to " << DataTypeName(x) << " is not implemented\n"; \
    __exit(-1);
#define UNREACHABLE()                  \
    std::cout << "Unreachable code\n"; \
    __exit(-1);
#define __exit(status)                        \
    {                                         \
        if (status != 0) {                    \
            std::cout << "Quantize failed\n"; \
            remove(output_path_.c_str());     \
        }                                     \
        exit(status);                         \
    }
static std::pair<void *, uint64_t> alloc_quant_block(uint64_t count, DataType type) {
    uint64_t size = DataTypeSize(type,count);
    if (size <= 0) {
        return std::make_pair(nullptr, 0);
    }
    void *data = new char[size];
    return std::make_pair(data, size);
}
namespace mllm {
class QuantWriter : public ParamWriter {
public:
    explicit QuantWriter(std::string output_path, std::string input_path);
    int ReadParams();
    void QuantParams(DataType dataType);

private:
    string output_path_;
    mllm::ParamLoader *param_loader_;
    DataType quant_type_;
    std::vector<std::string> param_names_;
    float *getParam(std::string param_name);
};
QuantWriter::QuantWriter(std::string output_path, std::string input_path) :
    ParamWriter(output_path), output_path_(output_path) {
    param_loader_ = new mllm::ParamLoader(std::move(input_path));
    if (param_loader_ == nullptr) {
        __exit(-1);
    }
}
int QuantWriter::ReadParams() {
    param_names_ = param_loader_->getParamNames();
    paddingIndex(param_names_);
    return param_names_.size();
}
float *QuantWriter::getParam(std::string param_name) {
    auto type = param_loader_->data_type_[param_name];
    if (type != DataType::MLLM_TYPE_F32) {
        return nullptr;
    }
    void *data = param_loader_->load(param_name);
    return static_cast<float *>(data);
}
void QuantWriter::QuantParams(DataType dataType) {
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
            std::cout<<name<<std::endl;
            if (name.find("norm") != std::string::npos) {
                auto s = param_loader_->offsets_[name].second / sizeof(float);
                auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
                writeParam(name, MLLM_TYPE_F32, param, tsize);
                std::cout<<"-----has norm-----"<<tsize<<std::endl;
            }
            else if (name.find("tok_embeddings") != std::string::npos){
                auto s = param_loader_->offsets_[name].second / sizeof(float);
                auto tsize = alloc_quant_block(s, MLLM_TYPE_F32).second;
                writeParam(name, MLLM_TYPE_F32, param, tsize);
                std::cout<<"-----has ebd-----"<<tsize<<std::endl;
            }else {
                writeParam(name, quant_type_, quant_ptr, size);
            }
            //writeParam(name, quant_type_, quant_ptr, size);
            delete[] (char *)quant_ptr;
        }
    }
    writeIndex();
}

} // namespace mllm
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: ./quantize <input_path> <output_path> <quant_type>\n";
        return -1;
    }
    auto input_path = std::string(argv[1]);
    auto output_path = std::string(argv[2]);
    auto quant_type = std::string(argv[3]);
    mllm::QuantWriter quant_writer(output_path, input_path);
    int param_count = quant_writer.ReadParams();
    if (param_count <= 0) {
        std::cout << "No params to quantize\n";
        return -1;
    }
    std::cout << "Quantize " << param_count << " params to " << quant_type << "\n";
    if (quant_type == "Q4_0") {
        quant_writer.QuantParams(MLLM_TYPE_Q4_0);
    } else if (quant_type == "Q8_0") {
        quant_writer.QuantParams(MLLM_TYPE_Q8_0);
    } else if (quant_type == "Q4_K") {
        quant_writer.QuantParams(MLLM_TYPE_Q4_K);
    } else if (quant_type == "Q8_K") {
        quant_writer.QuantParams(MLLM_TYPE_Q8_K);
    } else {
        std::cout << "Quant type " << quant_type << " is not supported\n";
        return -1;
    }
    return 0;
}