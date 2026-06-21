// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/CausalMaskOp.hpp"

namespace mllm::nn {

class CausalMask : public Layer {
 public:
  CausalMask();

  explicit CausalMask(const aops::CausalMaskOpOptions& options);

  CausalMask(bool sliding_window, int32_t window_size);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
