// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GraphOps.hpp"

namespace mllm::cpu {

CPUGraphBeginOp::CPUGraphBeginOp(const aops::GraphBeginOpOptions& options) : aops::GraphBeginOp(options) {}

CPUGraphEndOp::CPUGraphEndOp(const aops::GraphEndOpOptions& options) : aops::GraphEndOp(options) {}

}  // namespace mllm::cpu
