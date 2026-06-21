// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/nn/layers/SiLU.hpp"

namespace mllm::nn {

SiLU::SiLU() : Layer(OpTypes::kSiLU, aops::SiLUOpOptions{}) {}

SiLU::SiLU(const aops::SiLUOpOptions& options) : Layer(OpTypes::kSiLU, options) {}

}  // namespace mllm::nn
