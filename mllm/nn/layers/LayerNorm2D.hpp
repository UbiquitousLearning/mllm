// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/LayerNorm2DOp.hpp"

namespace mllm::nn {

class LayerNorm2D : public Layer {
 public:
  LayerNorm2D();

  explicit LayerNorm2D(const aops::LayerNorm2DOpOptions& options);

  explicit LayerNorm2D(const int32_t num_channels, float eps = 1e-6);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
  MLLM_LAYER_ENABLE_INPLACE_ATTRIBUTE(LayerNorm2D)
};

}  // namespace mllm::nn
