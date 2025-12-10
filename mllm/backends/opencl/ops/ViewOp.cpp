// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/opencl/ops/ViewOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::opencl {

OpenCLViewOp::OpenCLViewOp(const aops::ViewOpOptions& options) : aops::ViewOp(options) {}

}  // namespace mllm::opencl