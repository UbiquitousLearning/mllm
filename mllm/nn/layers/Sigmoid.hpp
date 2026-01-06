// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/SigmoidOp.hpp"

namespace mllm::nn {

class Sigmoid : public Layer {
 public:
  Sigmoid();

  explicit Sigmoid(const aops::SigmoidOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
  MLLM_LAYER_ENABLE_INPLACE_ATTRIBUTE(Sigmoid)
};

}  // namespace mllm::nn
