// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/Conv3DOp.hpp"

namespace mllm::cpu {

CPUConv3DOp::CPUConv3DOp(const aops::Conv3DOpOptions& options) : aops::Conv3DOp(options) {}

void CPUConv3DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

}  // namespace mllm::cpu
