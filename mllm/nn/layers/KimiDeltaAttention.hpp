// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/KimiDeltaAttentionOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class KimiDeltaAttention : public Layer {
 public:
  KimiDeltaAttention();
  explicit KimiDeltaAttention(const aops::KimiDeltaAttentionOpOptions& options);
  KimiDeltaAttention(bool safe_gate, float lower_bound, bool state_inplace = false);

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
