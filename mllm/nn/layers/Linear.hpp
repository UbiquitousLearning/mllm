// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::nn {

class Linear : public Layer {
 public:
  Linear();

  Linear(int32_t in_channels, int32_t out_channels, bool bias = true,
         aops::LinearImplTypes impl_type = aops::LinearImplTypes::kDefault);

  explicit Linear(const aops::LinearOpOptions& options);

  [[nodiscard]] Tensor weight() const;

  [[nodiscard]] Tensor bias() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
