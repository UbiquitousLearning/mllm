// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CmpOp.hpp"

namespace mllm::cpu {

class CPUEqualOp final : public aops::EqualOp {
 public:
  explicit CPUEqualOp(const aops::EqualOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUEqualOpFactory final : public TypedOpFactory<OpTypes::kEqual, aops::EqualOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::EqualOpOptions& options) override {
    return std::make_shared<CPUEqualOp>(options);
  }
};

}  // namespace mllm::cpu
