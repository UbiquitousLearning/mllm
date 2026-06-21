// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/LayerNorm2DOp.hpp"

namespace mllm::cpu {

class CPULayerNorm2DOp final : public aops::LayerNorm2DOp {
 public:
  explicit CPULayerNorm2DOp(const aops::LayerNorm2DOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPULayerNorm2DOpFactory : public TypedOpFactory<OpTypes::kLayerNorm2D, aops::LayerNorm2DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LayerNorm2DOpOptions& options) override {
    return std::make_shared<CPULayerNorm2DOp>(options);
  }
};

}  // namespace mllm::cpu
