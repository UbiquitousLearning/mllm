#pragma once

#include "mllm/models/ARGeneration.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Common.hpp"

#include <string>
#include <vector>

namespace mllm::qnn::aot {

class QnnAOTModule : public mllm::nn::Module, public models::ARGeneration {
 public:
  explicit QnnAOTModule(const std::string& graph_name);
  ~QnnAOTModule() {
    // Clear output tensors to ensure proper cleanup order
    output_tensors_.clear();
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override;

  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    NYI("ARGeneration forward is not implemented for QnnAOTModule");
    return {};
  };

  int64_t sampleGreedy(Tensor& logits);

  void setOutputTensors(const std::vector<Tensor>& output_tensors) { output_tensors_ = output_tensors; }

 private:
  std::string graph_name_;
  std::vector<Tensor> output_tensors_;
};

}  // namespace mllm::qnn::aot
