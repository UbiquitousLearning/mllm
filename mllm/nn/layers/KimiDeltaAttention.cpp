// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/KimiDeltaAttention.hpp"

namespace mllm::nn {

KimiDeltaAttention::KimiDeltaAttention() : Layer(OpTypes::kKimiDeltaAttention, aops::KimiDeltaAttentionOpOptions{}) {}

KimiDeltaAttention::KimiDeltaAttention(const aops::KimiDeltaAttentionOpOptions& options)
    : Layer(OpTypes::kKimiDeltaAttention, options) {}

KimiDeltaAttention::KimiDeltaAttention(bool safe_gate, float lower_bound, bool state_inplace)
    : Layer(OpTypes::kKimiDeltaAttention,
            aops::KimiDeltaAttentionOpOptions{
                .safe_gate = safe_gate, .lower_bound = lower_bound, .state_inplace = state_inplace}) {}

}  // namespace mllm::nn
