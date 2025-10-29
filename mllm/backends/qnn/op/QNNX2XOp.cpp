// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"
#include <cstring>

namespace mllm::qnn {

QNNX2XOp::QNNX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void QNNX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto& output = outputs[0];

  // For now, only do copy between CPU and QNN shared buffer
  // Ensure output tensor has allocated memory
  if (!output.impl()->storage()->ptr_) { output.alloc(); }

  // Get input data pointer
  const void* src_data = input.ptr<void>();
  void* dst_data = output.ptr<void>();

  // Calculate data size in bytes
  size_t data_size = input.bytes();

  // Perform memory copy from CPU to QNN shared buffer
  std::memcpy(dst_data, src_data, data_size);
}

bool QNNX2XPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  MLLM_ERROR_EXIT(1, "Illegal Modeling Arch, the tensor.to(kQNN/kCPU) should occur in QNN sub graph");

  return false;
}

}  // namespace mllm::qnn