// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/PatternSparseAttention.hpp"

namespace mllm::nn {

PatternSparseAttention::PatternSparseAttention() : PatternSparseAttention(aops::PatternSparseAttentionOpOptions{}) {}

PatternSparseAttention::PatternSparseAttention(const aops::PatternSparseAttentionOpOptions& options)
    : Layer(OpTypes::kPatternSparseAttention, options) {}

}  // namespace mllm::nn
