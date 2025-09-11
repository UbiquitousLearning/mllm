// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/MultimodalRoPEOp.hpp"

namespace mllm::nn {

class MultimodalRoPE : public Layer {
 public:
  MultimodalRoPE();

  explicit MultimodalRoPE(const aops::MultimodalRoPEOpOptions& options);

  explicit MultimodalRoPE(const aops::Qwen2VLMultimodalRoPEOpOptions& options,
                          aops::MultimodalRoPEOpOptionsInputType input_type = aops::MultimodalRoPEOpOptionsInputType::kBHSD);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
