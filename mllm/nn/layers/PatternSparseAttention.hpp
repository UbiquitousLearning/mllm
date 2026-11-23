// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/PatternSparseAttentionOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class PatternSparseAttention : public Layer {
 public:
  PatternSparseAttention();

  explicit PatternSparseAttention(const aops::PatternSparseAttentionOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
