// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/MatMulOp.hpp"

namespace mllm::cpu {

CPUMatMulOp::CPUMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {}

void CPUMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& lhs = inputs[0];
  auto& rhs = inputs[1];
  auto& o = outputs[0];

  // Only Support Contiguous Tensor
  MLLM_RT_ASSERT(lhs.isContiguous());
  MLLM_RT_ASSERT(rhs.isContiguous());

  auto transpose_a = options_.transpose_a;
  auto transpose_b = options_.transpose_b;

  // TODO
}

}  // namespace mllm::cpu
