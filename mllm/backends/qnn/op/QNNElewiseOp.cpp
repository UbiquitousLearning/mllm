// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNElewiseOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

QNNAddOp::QNNAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {}

void QNNAddOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Add operation expects 2 inputs and 1 output
  assert(inputs.size() == 2);

  const auto& input0 = inputs[0];
  const auto& input1 = inputs[1];

  // Get shapes
  auto shape0 = input0.shape();
  auto shape1 = input1.shape();

  // Check shapes are compatible for broadcasting
  if (shape0.size() != shape1.size()) {
    MLLM_ERROR("Input tensors must have same rank for Add operation");
    return;
  }

  for (size_t i = 0; i < shape0.size(); ++i) {
    if (shape0[i] != shape1[i] && shape0[i] != 1 && shape1[i] != 1) {
      MLLM_ERROR("Input tensors shapes are not compatible for broadcasting");
      return;
    }
  }

  // Output shape takes the maximum size in each dimension
  auto output_shape = shape0;
  for (size_t i = 0; i < shape0.size(); ++i) { output_shape[i] = std::max(shape0[i], shape1[i]); }

  outputs.emplace_back(Tensor::empty(output_shape, input0.dtype(), input0.device()));
}

bool QNNAddPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the AddOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::AddOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::AddOp");
    return false;
  }

  auto* qnnAddOp = dynamic_cast<QNNAddOp*>(linalgOp->getAOp());
  if (!qnnAddOp) {
    MLLM_ERROR("Failed to cast to QNNAddOp");
    return false;
  }

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend from context");
    return false;
  }

  // Prepare input tensor names (should be 2 inputs for Add operation)
  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add Add node to the graph using LLaMAPackage
  qnnBackend->graphAddNode(graphName, qnnAddOp->getName(), "LLaMAAdd", inputTensorNames, {outputs[0]->name()}, {}, {},
                           QNN_Custom_Op_Package);

  return true;
}

}  // namespace mllm::qnn
