// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/CausalMaskOp.hpp"
#include "mllm/nn/layers/CausalMask.hpp"

namespace mllm::nn {

CausalMask::CausalMask() : Layer(OpTypes::kCausalMask, aops::CausalMaskOpOptions{}) {}

CausalMask::CausalMask(const aops::CausalMaskOpOptions& options) : Layer(OpTypes::kCausalMask, options) {}

CausalMask::CausalMask(bool sliding_window, int32_t window_size)
    : Layer(OpTypes::kCausalMask, aops::CausalMaskOpOptions{.sliding_window = sliding_window, .window_size = window_size}) {}

}  // namespace mllm::nn
