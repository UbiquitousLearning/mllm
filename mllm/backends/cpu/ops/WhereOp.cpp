// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <vector>

#include "mllm/backends/cpu/ops/WhereOp.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::cpu {

CPUWhereOp::CPUWhereOp(const aops::WhereOpOptions& options) : aops::WhereOp(options) {}

void CPUWhereOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_RT_ASSERT(false); }

}  // namespace mllm::cpu
