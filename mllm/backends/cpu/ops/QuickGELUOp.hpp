// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/QuickGELUOp.hpp"

namespace mllm::cpu {

class CPUQuickGELUOp final : public aops::QuickGELUOp {
 public:
  explicit CPUQuickGELUOp(const aops::QuickGELUOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUQuickGELUOpFactory : public TypedOpFactory<OpTypes::kQuickGELU, aops::QuickGELUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::QuickGELUOpOptions& options) override {
    return std::make_shared<CPUQuickGELUOp>(options);
  }
};

}  // namespace mllm::cpu
