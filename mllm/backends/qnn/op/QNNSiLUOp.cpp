// Copyright (c) MLLM Team.
// Licensed under the MIT License.

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

  // Add SiLU node to the graph using custom QNN operator from LLaMAPackage
  qnnBackend->graphAddNode(graphName, qnnSiLUOp->getName(), "SiLU", inputTensorNames, {outputs[0]->name()}, {}, {},
                           "LLaMAPackage");

  return true;
}

}  // namespace mllm::qnn
