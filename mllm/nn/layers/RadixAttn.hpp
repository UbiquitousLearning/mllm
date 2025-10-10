// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/RadixAttnOp.hpp"

namespace mllm::nn {

class RadixAttn : public Layer {
 public:
  RadixAttn();

  explicit RadixAttn(const aops::RadixAttnOpOptions& options);

  RadixAttn(int32_t H_Q, int32_t H_KV);

  // Q, K, V in and one output out
  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
