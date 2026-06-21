// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SigmoidOp.hpp"
#include "mllm/nn/layers/Sigmoid.hpp"

namespace mllm::nn {

Sigmoid::Sigmoid() : Layer(OpTypes::kSigmoid, aops::SigmoidOpOptions{}) {}

Sigmoid::Sigmoid(const aops::SigmoidOpOptions& options) : Layer(OpTypes::kSigmoid, options) {}

}  // namespace mllm::nn
