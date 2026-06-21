// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/TransposeOp.hpp"

namespace mllm::cpu {

class CPUTransposeOp final : public aops::TransposeOp {
 public:
  explicit CPUTransposeOp(const aops::TransposeOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUTransposeOpFactory final : public TypedOpFactory<OpTypes::kTranspose, aops::TransposeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TransposeOpOptions& options) override {
    return std::make_shared<CPUTransposeOp>(options);
  }
};

}  // namespace mllm::cpu