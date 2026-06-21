// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNTransposeOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"
#include "QnnTypes.h"
#include <cstring>

namespace mllm::qnn {

QNNTransposeOp::QNNTransposeOp(const aops::TransposeOpOptions& options) : aops::TransposeOp(options) {}

void QNNTransposeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto input_shape = input.shape();

  const auto& options = this->options();

  // For simple 2D transpose, swap dim0 and dim1
  auto output_shape = input_shape;
  if (options.dim0 < input_shape.size() && options.dim1 < input_shape.size()) {
    std::swap(output_shape[options.dim0], output_shape[options.dim1]);
  }

  // Transpose output has the permuted shape of input
  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));

  // Propagate quantization scale from input to output
  propagateQuantScale(inputs[0], outputs[0]);
}

bool QNNTransposePattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                                  const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                  const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the TransposeOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::TransposeOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::TransposeOp");
    return false;
  }

  auto* qnnTransposeOp = dynamic_cast<QNNTransposeOp*>(linalgOp->getAOp());
  if (!qnnTransposeOp) {
    MLLM_ERROR("Failed to cast to QNNTransposeOp");
    return false;
  }

  const auto& options = qnnTransposeOp->options();

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) { MLLM_ERROR("Failed to get QNN backend from context"); }

  // Prepare input tensor names
  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  // Create quantization parameters using helper function
  auto quantParam = createQuantizeParams(inputs[0]->tensor_);
  if (inputs[0]->tensor_.dtype() == kInt8 || inputs[0]->tensor_.dtype() == kInt16) {
    MLLM_INFO("quant scale in transpose {}", getQuantScale(outputs[0]->tensor_));
  }

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_, quantParam)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Create perm parameter for QNN Transpose
  // For simple 2D transpose, create permutation array based on dim0 and dim1
  const auto& input_tensor = inputs[0]->tensor_;
  auto input_shape = input_tensor.shape();
  int rank = input_shape.size();

  // Initialize perm with identity permutation
  std::vector<uint32_t> perm(rank);
  for (int i = 0; i < rank; ++i) { perm[i] = i; }

  // Swap the specified dimensions
  if (options.dim0 < rank && options.dim1 < rank) { std::swap(perm[options.dim0], perm[options.dim1]); }

  // Create QNN parameter for permutation using QNNParamTensorWrapper
  auto permParam = mllm::qnn::QNNParamTensorWrapper::create("perm", qnnTransposeOp->getName() + ".perm", QNN_DATATYPE_UINT_32,
                                                            std::vector<int32_t>{rank});

  // Set the permutation data
  uint32_t* permData = reinterpret_cast<uint32_t*>(permParam->alloc());
  std::memcpy(permData, perm.data(), rank * sizeof(uint32_t));

  // Add Transpose node to the graph
  qnnBackend->graphAddNode(graphName, qnnTransposeOp->getName(), "Transpose", inputTensorNames, {outputs[0]->name()},
                           {permParam}, {});

  return true;
}

}  // namespace mllm::qnn
