// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/ReLUOp.hpp"

namespace mllm::nn {

class ReLU : public Layer {
 public:
  ReLU();

  explicit ReLU(const aops::ReLUOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
