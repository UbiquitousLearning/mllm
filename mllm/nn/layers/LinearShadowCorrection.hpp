// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearShadowCorrectionOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class LinearShadowCorrection : public Layer {
 public:
  LinearShadowCorrection();

  LinearShadowCorrection(int32_t in_channels, int32_t out_channels, int32_t activation_bits = 16);

  explicit LinearShadowCorrection(const aops::LinearShadowCorrectionOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
