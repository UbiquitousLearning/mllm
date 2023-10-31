//
// Created by lx on 23-10-31.
//
#define NOT_IMPLEMENTED(x)                                               \
    std::cout << "Quantize params to " << #x << " is not implemented\n"; \
    exit(-1);
#define UNREACHABLE()                  \
    std::cout << "Unreachable code\n"; \
    exit(-1);
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include <string>
namespace mllm {
class QuantWriter : public ParamWriter {
    explicit QuantWriter(std::string output_path, std::string input_path);
    int ReadParams();
    void QuantParams(DataType dataType);

private:
    mllm::ParamLoader *param_loader_;
    DataType quant_type_;
    std::vector<std::string> param_names_;
    float *getParam(std::string param_name);
};
QuantWriter::QuantWriter(std::string output_path, std::string input_path) :
    ParamWriter(std::move(output_path)) {
    param_loader_ = new mllm::ParamLoader(std::move(input_path));
    if (param_loader_ == nullptr) {
        exit(-1);
    }
}
int QuantWriter::ReadParams() {
    param_names_ = param_loader_->getParamNames();
    paddingIndex(param_names_);
    return param_names_.size();
}
float *QuantWriter::getParam(std::string param_name) {
    auto type = param_loader_->data_type_[param_name];
    if (type != mllm::DataType::FP32) {
        return nullptr;
    }
    void *data = param_loader_->load(param_name);
    return static_cast<float *>(data);
}
void QuantWriter::QuantParams(DataType dataType) {
    quant_type_ = dataType;
    for (const auto &name : param_names_) {
        auto *param = getParam(name);
        if (param == nullptr) {
            exit(-1);
        }
        switch (dataType) {
        case FP32:
            std::cout << "No need to quantize FP32 params\n";
            break;
        case FP16:
            NOT_IMPLEMENTED(FP16);
            break;

        case INT8: break;
        case INT4:

            break;
        case DATA_TYPE_COUNT: UNREACHABLE(); break;
        }
    }
}

} // namespace mllm
