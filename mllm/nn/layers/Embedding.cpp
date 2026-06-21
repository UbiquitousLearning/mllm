// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/EmbeddingOp.hpp"
#include "mllm/nn/layers/Embedding.hpp"

namespace mllm::nn {

Embedding::Embedding() : Layer(OpTypes::kEmbedding, aops::EmbeddingOpOptions{}) {}

Embedding::Embedding(const aops::EmbeddingOpOptions& options) : Layer(OpTypes::kEmbedding, options) {}

Embedding::Embedding(int32_t vocab_size, int32_t hidden_size)
    : Layer(OpTypes::kEmbedding, aops::EmbeddingOpOptions{.vocab_size = vocab_size, .hidden_size = hidden_size}) {}

Tensor Embedding::weight() const { return std::static_pointer_cast<aops::EmbeddingOp>(impl()->getInstancedOp())->weight(); }

}  // namespace mllm::nn
