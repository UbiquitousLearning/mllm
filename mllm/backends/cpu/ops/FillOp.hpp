// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/FillOp.hpp"

namespace mllm::cpu {

class CPUFillOp final : public aops::FillOp {
 public:
  explicit CPUFillOp(const aops::FillOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUFillOpFactory final : public TypedOpFactory<OpTypes::kFill, aops::FillOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FillOpOptions& options) override {
    return std::make_shared<CPUFillOp>(options);
  }
};

}  // namespace mllm::cpu
