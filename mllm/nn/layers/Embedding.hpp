// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"

namespace mllm::nn {

class Embedding : public Layer {
 public:
  Embedding();

  explicit Embedding(const aops::EmbeddingOpOptions& options);

  Embedding(int32_t vocab_size, int32_t hidden_size);

  [[nodiscard]] Tensor weight() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
