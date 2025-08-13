// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"

namespace mllm::nn {

class Softmax : public Layer {
 public:
  Softmax();

  explicit Softmax(const aops::SoftmaxOpOptions& options);

  explicit Softmax(int32_t dim);
};

}  // namespace mllm::nn
