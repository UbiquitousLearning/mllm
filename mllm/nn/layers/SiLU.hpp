// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/SiLUOp.hpp"

namespace mllm::nn {

class SiLU : public Layer {
 public:
  SiLU();

  explicit SiLU(const aops::SiLUOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
  MLLM_LAYER_ENABLE_INPLACE_ATTRIBUTE(SiLU)
};

}  // namespace mllm::nn
