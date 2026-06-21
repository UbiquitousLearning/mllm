// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNViewOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"
#include <cstring>
#include <numeric>

namespace mllm::qnn {

QNNViewOp::QNNViewOp(const aops::ViewOpOptions& options) : aops::ViewOp(options) {}

void QNNViewOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  const auto& input_shape = input.shape();
  const auto& target_shape = this->options().to_shape;

  // Calculate the inferred shape
  std::vector<int32_t> actual_shape = target_shape;
  int infer_dim = -1;
  size_t product = 1;

  for (size_t i = 0; i < actual_shape.size(); ++i) {
    if (actual_shape[i] == -1) {
      if (infer_dim != -1) {
        MLLM_ERROR("Only one dimension can be inferred (-1)");
        return;
      }
      infer_dim = static_cast<int>(i);
    } else if (actual_shape[i] == 0) {
      // 0 means copy the corresponding dimension from input
      if (i < input_shape.size()) {
        actual_shape[i] = input_shape[i];
      } else {
        MLLM_ERROR("Invalid 0 dimension at index {} for input shape size {}", i, input_shape.size());
        return;
      }
      product *= actual_shape[i];
    } else {
      if (actual_shape[i] < 0) {
        MLLM_ERROR("Shape dimensions must be >= -1, got {}", actual_shape[i]);
        return;
      }
      product *= actual_shape[i];
    }
  }

  // Infer the -1 dimension
  const int64_t input_numel = input.numel();
  if (infer_dim != -1) {
    if (product == 0) {
      MLLM_ERROR("Cannot infer dimension for a shape with zero product");
      return;
    }
    if (input_numel % product != 0) {
      MLLM_ERROR("Input tensor size {} does not match inferred shape product {}", input_numel, product);
      return;
    }
    actual_shape[infer_dim] = static_cast<int32_t>(input_numel / product);
  }

  // Verify total element count matches
  const int64_t new_numel = std::accumulate(actual_shape.begin(), actual_shape.end(), 1, std::multiplies<>());
  if (input_numel != new_numel) {
    MLLM_ERROR("Input tensor size {} does not match output size {}", input_numel, new_numel);
    return;
  }

  // Create output tensor with the new shape
  outputs.emplace_back(Tensor::empty(actual_shape, input.dtype(), input.device()));

  // Propagate quantization scale from input to output
  propagateQuantScale(inputs[0], outputs[0]);
}

bool QNNViewPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                             const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                             const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the ViewOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::ViewOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::ViewOp");
    return false;
  }

  auto* qnnViewOp = dynamic_cast<QNNViewOp*>(linalgOp->getAOp());
  if (!qnnViewOp) {
    MLLM_ERROR("Failed to cast to QNNViewOp");
    return false;
  }

  const auto& options = qnnViewOp->options();

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend from context");
    return false;
  }

  // Prepare input tensor names
  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  // Create quantization parameters using helper function
  auto quantParam = createQuantizeParams(outputs[0]->tensor_);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_, quantParam)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add Reshape node to the graph without shape parameter (identity reshape)
  qnnBackend->graphAddNode(graphName, qnnViewOp->getName(), "Reshape", inputTensorNames, {outputs[0]->name()}, {}, {});

  return true;
}

}  // namespace mllm::qnn
