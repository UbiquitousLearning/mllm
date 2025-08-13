// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/SliceOp.hpp"

namespace mllm::cpu {

CPUSliceOp::CPUSliceOp(const aops::SliceOpOptions& options) : aops::SliceOp(options) {}

}  // namespace mllm::cpu
