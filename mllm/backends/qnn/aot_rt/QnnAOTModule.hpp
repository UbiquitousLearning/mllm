#pragma once

#include "mllm/models/ARGeneration.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Common.hpp"

#include <string>
#include <vector>

namespace mllm::qnn::aot {

class QnnAOTModule : public mllm::nn::Module, public models::ARGeneration {
 public:
  QnnAOTModule(const std::string& model_path, const std::string& graph_name);

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override;

  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    NYI("ARGeneration forward is not implemented for QnnAOTModule");
    return {};
  };

  void setOutputTensors(const std::vector<Tensor>& output_tensors) { output_tensors_ = output_tensors; }

 private:
  std::string model_path_;
  std::string graph_name_;

  std::vector<Tensor> output_tensors_;

  std::string backend_path_;
};

}  // namespace mllm::qnn::aot
