// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/ParamOp.hpp"

namespace mllm::nn {

class Param : public Layer {
 public:
  Param();

  explicit Param(const aops::ParamOpOptions& options);

  explicit Param(const std::string& name, const Tensor::shape_t& shape = {});

  [[nodiscard]] Tensor weight() const;

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
