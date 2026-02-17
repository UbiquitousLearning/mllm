#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include <algorithm>

namespace mllm::qnn::aot {

QnnAOTModule::QnnAOTModule(const std::string& graph_name) : mllm::nn::Module(graph_name), graph_name_(graph_name) {}

std::vector<mllm::Tensor> QnnAOTModule::forward(const std::vector<mllm::Tensor>& inputs,
                                                const std::vector<mllm::AnyValue>& args) {
  return output_tensors_;
}

int64_t QnnAOTModule::sampleGreedy(mllm::Tensor& logits) {
  auto logits_data = logits.ptr<uint16_t>();
  int vocab_size = logits.shape().back();
  auto max_it = std::max_element(logits_data, logits_data + vocab_size);
  return std::distance(logits_data, max_it);
}

}  // namespace mllm::qnn::aot
