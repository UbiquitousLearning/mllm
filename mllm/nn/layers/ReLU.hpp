// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/ReLUOp.hpp"

namespace mllm::nn {

class ReLU : public Layer {
 public:
  ReLU();

  explicit ReLU(const aops::ReLUOpOptions& options);
};

}  // namespace mllm::nn
