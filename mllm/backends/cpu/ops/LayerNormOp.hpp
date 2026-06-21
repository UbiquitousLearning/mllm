// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/LayerNormOp.hpp"

namespace mllm::cpu {

class CPULayerNormOp final : public aops::LayerNormOp {
 public:
  explicit CPULayerNormOp(const aops::LayerNormOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPULayerNormOpFactory : public TypedOpFactory<OpTypes::kLayerNorm, aops::LayerNormOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LayerNormOpOptions& options) override {
    return std::make_shared<CPULayerNormOp>(options);
  }
};

}  // namespace mllm::cpu
