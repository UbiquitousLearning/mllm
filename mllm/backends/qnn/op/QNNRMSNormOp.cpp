// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNRMSNormOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

QNNRMSNormOp::QNNRMSNormOp(const aops::RMSNormOpOptions& options) : aops::RMSNormOp(options) {}

void QNNRMSNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto input_shape = input.shape();

  // RMSNorm output has the same shape as input
  outputs.emplace_back(Tensor::empty(input_shape, input.dtype(), input.device()));
}

bool QNNRMSNormPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                                const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the RMSNormOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::RMSNormOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::RMSNormOp");
    return false;
  }

  auto* qnnRMSNormOp = dynamic_cast<QNNRMSNormOp*>(linalgOp->getAOp());
  if (!qnnRMSNormOp) {
    MLLM_ERROR("Failed to cast to QNNRMSNormOp");
    return false;
  }

  const auto& options = qnnRMSNormOp->options();

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) { MLLM_ERROR("Failed to get QNN backend from context"); }

  auto& weight = qnnRMSNormOp->weight();

  // Prepare input tensor names (input + weight)
  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }
  inputTensorNames.push_back(weight.name());

  if (!qnnBackend->addStaticTensor(graphName, weight.name(), weight)) {
    MLLM_ERROR("Failed to add weight tensor {} to graph {}", weight.name(), graphName);
    return false;
  }

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add RMSNorm node to the graph using LLaMAPackage
  qnnBackend->graphAddNode(graphName, qnnRMSNormOp->getName(), "RMSNorm", inputTensorNames, {outputs[0]->name()}, {}, {},
                           QNN_Custom_Op_Package);

  return true;
}

}  // namespace mllm::qnn
