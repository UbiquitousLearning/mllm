// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/QuickGELUOp.hpp"
#include "mllm/nn/layers/QuickGELU.hpp"

namespace mllm::nn {

QuickGELU::QuickGELU() : Layer(OpTypes::kQuickGELU, aops::QuickGELUOpOptions{}) {}

QuickGELU::QuickGELU(const aops::QuickGELUOpOptions& options) : Layer(OpTypes::kQuickGELU, options) {}

}  // namespace mllm::nn
