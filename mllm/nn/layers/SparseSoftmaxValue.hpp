// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/SparseSoftmaxValueOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

class SparseSoftmaxValue : public Layer {
 public:
  SparseSoftmaxValue();
  explicit SparseSoftmaxValue(const aops::SparseSoftmaxValueOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
