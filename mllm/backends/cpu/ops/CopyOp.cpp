// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/CopyOp.hpp"
#include <cstring>

namespace mllm::cpu {

CPUCopyOp::CPUCopyOp(const aops::CopyOpOptions& options) : aops::CopyOp(options) {}

void CPUCopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i0 = inputs[0];
  auto& i1 = inputs[1];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(i0.isContiguous());

  std::memcpy(i1.ptr<char>(), i0.ptr<char>(), i0.bytes());
}

}  // namespace mllm::cpu
