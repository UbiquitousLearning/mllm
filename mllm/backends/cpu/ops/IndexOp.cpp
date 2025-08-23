// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/IndexOp.hpp"

namespace mllm::cpu {

CPUIndexOp::CPUIndexOp(const aops::IndexOpOptions& options) : aops::IndexOp(options) {}

}  // namespace mllm::cpu
