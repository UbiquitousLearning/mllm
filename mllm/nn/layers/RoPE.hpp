// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/RoPEOp.hpp"

namespace mllm::nn {

class RoPE : public Layer {
 public:
  RoPE();

  RoPE(float theta, int32_t max_position_embeddings,
       aops::RoPEOpOptionsInputType input_type = aops::RoPEOpOptionsInputType::kBHSD);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD

 private:
  float rope_theta_ = 10000.0F;
  int32_t max_position_embeddings_ = 128;
};

}  // namespace mllm::nn
