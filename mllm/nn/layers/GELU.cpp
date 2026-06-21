// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/GELUOp.hpp"
#include "mllm/nn/layers/GELU.hpp"

namespace mllm::nn {

GELU::GELU() : Layer(OpTypes::kGELU, aops::GELUOpOptions{}) {}

GELU::GELU(const aops::GELUOpOptions& options) : Layer(OpTypes::kGELU, options) {}

}  // namespace mllm::nn
