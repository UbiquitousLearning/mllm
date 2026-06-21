// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"

namespace mllm::cpu {

class CPUMatMulOp final : public aops::MatMulOp {
 public:
  explicit CPUMatMulOp(const aops::MatMulOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUMatMulOpFactory : public TypedOpFactory<OpTypes::kMatMul, aops::MatMulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MatMulOpOptions& options) override {
    return std::make_shared<CPUMatMulOp>(options);
  }
};

}  // namespace mllm::cpu
