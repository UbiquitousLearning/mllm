// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CausalMaskOp.hpp"

namespace mllm::cpu {

class CPUCausalMaskOp final : public aops::CausalMaskOp {
 public:
  explicit CPUCausalMaskOp(const aops::CausalMaskOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUCausalMaskOpFactory : public TypedOpFactory<OpTypes::kCausalMask, aops::CausalMaskOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CausalMaskOpOptions& options) override {
    return std::make_shared<CPUCausalMaskOp>(options);
  }
};

}  // namespace mllm::cpu
