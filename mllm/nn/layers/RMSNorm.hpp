// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"

namespace mllm::nn {

class RMSNorm : public Layer {
 public:
  RMSNorm();

  explicit RMSNorm(float epsilon, bool add_unit_offset = false);

  explicit RMSNorm(const aops::RMSNormOpOptions& options);

  [[nodiscard]] Tensor weight() const;
};

}  // namespace mllm::nn
