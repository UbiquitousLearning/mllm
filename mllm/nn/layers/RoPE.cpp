// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/RoPE.hpp"
#include "mllm/core/aops/RoPEOp.hpp"

namespace mllm::nn {

RoPE::RoPE() : Layer(OpTypes::kRoPE, aops::RoPEOpOptions{}) {}

RoPE::RoPE(float theta, int32_t max_position_embeddings)
    : Layer(OpTypes::kRoPE, aops::RoPEOpOptions(theta, max_position_embeddings)) {}

}  // namespace mllm::nn