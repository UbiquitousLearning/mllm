#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"

namespace mllm::qnn::aot {

QnnAOTModule::QnnAOTModule(const std::string& model_path, const std::string& graph_name)
    : mllm::nn::Module(graph_name), model_path_(model_path), graph_name_(graph_name) {}

std::vector<mllm::Tensor> QnnAOTModule::forward(const std::vector<mllm::Tensor>& inputs,
                                                const std::vector<mllm::AnyValue>& args) {
  return output_tensors_;
}

}  // namespace mllm::qnn::aot
