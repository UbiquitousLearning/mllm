// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/GELUOp.hpp"

namespace mllm::nn {

class GELU : public Layer {
 public:
  GELU();

  explicit GELU(const aops::GELUOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
