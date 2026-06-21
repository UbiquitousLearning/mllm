// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/LayerNormOp.hpp"

namespace mllm::nn {

class LayerNorm : public Layer {
 public:
  LayerNorm();

  explicit LayerNorm(const aops::LayerNormOpOptions& options);

  explicit LayerNorm(const std::vector<int32_t>& normalized_shape, bool elementwise_affine = true, bool bias = true,
                     float eps = 1e-6);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
