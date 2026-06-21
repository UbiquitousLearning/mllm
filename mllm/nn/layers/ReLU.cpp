// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ReLUOp.hpp"
#include "mllm/nn/layers/ReLU.hpp"

namespace mllm::nn {

ReLU::ReLU() : Layer(OpTypes::kReLU, aops::ReLUOpOptions{}) {}

ReLU::ReLU(const aops::ReLUOpOptions& options) : Layer(OpTypes::kReLU, options) {}

}  // namespace mllm::nn
