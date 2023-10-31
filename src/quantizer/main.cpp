//
// Created by lx on 23-10-31.
//
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include <string>
namespace mllm {
class QuantWriter : public ParamWriter {
    explicit QuantWriter(std::string output_path, std::string input_path);
    int ReadParams();

private:
    mllm::ParamLoader *param_loader_;
    std::vector<std::string> param_names_;
    float *GetParam(std::string param_name);
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
float *QuantWriter::GetParam(std::string param_name) {
}

} // namespace mllm
