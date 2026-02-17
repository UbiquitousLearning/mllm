// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendMatMulOp final : public aops::MatMulOp {
 public:
  explicit AscendMatMulOp(const aops::MatMulOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendMatMulOpFactory : public TypedOpFactory<OpTypes::kMatMul, aops::MatMulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MatMulOpOptions& options) override {
    return std::make_shared<AscendMatMulOp>(options);
  }
};

}  // namespace mllm::ascend