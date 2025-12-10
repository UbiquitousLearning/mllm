// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/GraphOps.hpp"

namespace mllm::opencl {

OpenCLGraphBeginOp::OpenCLGraphBeginOp(const aops::GraphBeginOpOptions& options) : aops::GraphBeginOp(options) {}

OpenCLGraphEndOp::OpenCLGraphEndOp(const aops::GraphEndOpOptions& options) : aops::GraphEndOp(options) {}

}  // namespace mllm::opencl
