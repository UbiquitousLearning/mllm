// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/CloneOp.hpp"

namespace mllm::cpu {

CPUCloneOp::CPUCloneOp(const aops::CloneOpOptions& options) : aops::CloneOp(options) {}

void CPUCloneOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];
  auto& o = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(i.isContiguous());

  std::memcpy(o.ptr<char>(), i.ptr<char>(), i.bytes());
}

}  // namespace mllm::cpu
