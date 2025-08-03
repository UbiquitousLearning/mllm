// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"

namespace mllm::cpu {

class CPUX2XOp final : public aops::X2XOp {
 public:
  explicit CPUX2XOp(const aops::X2XOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUX2XOpFactory : public TypedOpFactory<OpTypes::kX2X, aops::X2XOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::X2XOpOptions& options) override {
    return std::make_shared<CPUX2XOp>(options);
  }
};

}  // namespace mllm::cpu
