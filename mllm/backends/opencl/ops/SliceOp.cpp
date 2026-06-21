// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/opencl/ops/SliceOp.hpp"

namespace mllm::opencl {

OpenCLSliceOp::OpenCLSliceOp(const aops::SliceOpOptions& options) : aops::SliceOp(options) {}

}  // namespace mllm::opencl