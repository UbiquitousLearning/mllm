// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/Tanh.hpp"

namespace mllm::nn {

Tanh::Tanh() : Layer(OpTypes::kTanh, aops::TanhOpOptions{}) {}

Tanh::Tanh(const aops::TanhOpOptions& options) : Layer(OpTypes::kTanh, options) {}

}  // namespace mllm::nn
