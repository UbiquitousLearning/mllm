// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "AvgPool1dOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

AvgPool1dOp::AvgPool1dOp(const AvgPool1dOpOptions& options)
    : BaseOp(OpTypes::kAvgPool1d), options_(options) {}

void AvgPool1dOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  if (ishape.size() != 3) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "AvgPool1dOp expects 3D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  const int batch = ishape[0];
  const int channels = ishape[1];
  const int length = ishape[2];

  const int kernel_size = options_.kernel_size;
  const int stride = options_.stride;
  const int padding = options_.padding;

  int pooled_length;
  if (options_.ceil_mode) {
    pooled_length = static_cast<int>(std::ceil(static_cast<float>(length + 2 * padding - kernel_size) / stride)) + 1;
  } else {
    pooled_length = ((length + 2 * padding - kernel_size) / stride) + 1;
  }

  auto new_shape = std::vector<int32_t>{batch, channels, pooled_length};

  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void AvgPool1dOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AvgPool1dOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Forward will be dispatched to backend implementation
}

}  // namespace mllm::aops
