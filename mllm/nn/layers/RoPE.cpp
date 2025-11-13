// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/RoPE.hpp"
#include "mllm/core/aops/RoPEOp.hpp"

namespace mllm::nn {

RoPE::RoPE() : Layer(OpTypes::kRoPE, aops::RoPEOpOptions{}) {}

RoPE::RoPE(float theta, int32_t max_position_embeddings, aops::RoPEOpOptionsInputType input_type)
    : Layer(OpTypes::kRoPE,
            aops::RoPEOpOptions{
                .rope_theta = theta, .max_position_embeddings = max_position_embeddings, .input_type = input_type}) {}

RoPE::RoPE(float theta, int32_t max_position_embeddings, int32_t partial_dim, aops::RoPEOpOptionsInputType input_type)
    : Layer(OpTypes::kRoPE, aops::RoPEOpOptions{.rope_theta = theta,
                                                .max_position_embeddings = max_position_embeddings,
                                                .input_type = input_type,
                                                .partial_dim = partial_dim}) {}
}  // namespace mllm::nn
