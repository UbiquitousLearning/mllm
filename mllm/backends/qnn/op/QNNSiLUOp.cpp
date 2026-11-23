// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/ShadowCorrectionGraph.hpp"
#include <cstdlib>
#include <string_view>

#include "mllm/backends/qnn/op/QNNSiLUOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

QNNSiLUOp::QNNSiLUOp(const aops::SiLUOpOptions& options) : aops::SiLUOp(options) {}

void QNNSiLUOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // SiLU operation expects 1 input and 1 output
  assert(inputs.size() == 1);

  const auto& input = inputs[0];

  // SiLU is an element-wise operation, so output shape is the same as input shape
  auto output_shape = input.shape();

  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

bool QNNSiLUPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                             const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                             const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the SiLUOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::SiLUOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::SiLUOp");
    return false;
  }

  auto* qnnSiLUOp = dynamic_cast<QNNSiLUOp*>(linalgOp->getAOp());
  if (!qnnSiLUOp) {
    MLLM_ERROR("Failed to cast to QNNSiLUOp");
    return false;
  }

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend from context");
    return false;
  }

  // Prepare input tensor names (should be 1 input for SiLU operation)
  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  const auto& node_name = qnnSiLUOp->getName();
  const auto graph = shadowMainGraphName(graphName);
  constexpr std::string_view layer_prefix = "model.layers.";
  const char* request = std::getenv("MLLM_QNN_QWEN_FP32_HVX_SILU");
  bool use_hvx = false;
  if (qnnSiLUOp->options().high_precision && request && std::string_view(request) == "1"
      && graph.starts_with(layer_prefix) && (graph.ends_with("_1") || graph.ends_with("_2"))
      && inputs.size() == 1 && outputs.size() == 1) {
    // Only change the FP32 gate activation of Qwen 1.5B at the compiled M256 shape.
    const auto layer_name = graph.substr(0, graph.size() - 2);
    const auto layer = layer_name.substr(layer_prefix.size());
    const auto& shape = inputs[0]->tensor_.shape();
    const bool supported_tile = shape == Tensor::shape_t{1, 256, 1, 8960} || shape == Tensor::shape_t{1, 16, 16, 8960};
    use_hvx = supported_tile && !layer.empty() && layer.find_first_not_of("0123456789") == std::string_view::npos
              && node_name == std::string(layer_name) + ".mlp.act" && inputs[0]->tensor_.dtype() == kFloat32
              && outputs[0]->tensor_.dtype() == kFloat32 && outputs[0]->tensor_.shape() == shape;
  }
  if (qnnSiLUOp->options().high_precision && !use_hvx) {
    const auto sigmoidTensorName = qnnSiLUOp->getName() + ".sigmoid";
    if (!qnnBackend->addTensor(graphName, sigmoidTensorName, QNN_TENSOR_TYPE_NATIVE, outputs[0]->tensor_)) {
      MLLM_ERROR("Failed to add high-precision SiLU intermediate tensor {} to graph {}", sigmoidTensorName, graphName);
      return false;
    }
    qnnBackend->graphAddNode(graphName, qnnSiLUOp->getName() + ".sigmoid", "Sigmoid", inputTensorNames, {sigmoidTensorName}, {},
                             {});
    qnnBackend->graphAddNode(graphName, qnnSiLUOp->getName(), "LLaMAMul", {sigmoidTensorName, inputs[0]->name()},
                             {outputs[0]->name()}, {}, {}, "LLaMAPackage");
  } else {
    // This custom operator evaluates SiLU with the package's polynomial approximation.
    qnnBackend->graphAddNode(graphName, node_name, "SiLU", inputTensorNames, {outputs[0]->name()}, {}, {},
                            "LLaMAPackage");
    if (use_hvx) { MLLM_INFO("QNN_MLP_FP32_HVX_SILU_LOWERING graph={} node={}", graphName, node_name); }
  }

  return true;
}

}  // namespace mllm::qnn
