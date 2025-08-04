// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/QuickGELUOp.hpp"

namespace mllm::nn {

class QuickGELU : public Layer {
 public:
  QuickGELU();

  explicit QuickGELU(const aops::QuickGELUOpOptions& options);
};

}  // namespace mllm::nn