// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/TanhOp.hpp"

namespace mllm::nn {

class Tanh : public Layer {
 public:
  Tanh();

  explicit Tanh(const aops::TanhOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
  MLLM_LAYER_ENABLE_INPLACE_ATTRIBUTE(Tanh)
};

}  // namespace mllm::nn
