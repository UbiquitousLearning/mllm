// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/ParamOp.hpp"

namespace mllm::cpu {

CPUParamOp::CPUParamOp(const aops::ParamOpOptions& options) : aops::ParamOp(options) {}

}  // namespace mllm::cpu
