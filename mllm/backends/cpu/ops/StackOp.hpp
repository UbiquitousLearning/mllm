// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/StackOp.hpp"

namespace mllm::cpu {

class CPUStackOp final : public aops::StackOp {
 public:
  explicit CPUStackOp(const aops::StackOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUStackOpFactory : public TypedOpFactory<OpTypes::kStack, aops::StackOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::StackOpOptions& options) override {
    return std::make_shared<CPUStackOp>(options);
  }
};

}  // namespace mllm::cpu