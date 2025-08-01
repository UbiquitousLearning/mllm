// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/PermuteOp.hpp"

namespace mllm::cpu {

class CPUPermuteOp final : public aops::PermuteOp {
 public:
  explicit CPUPermuteOp(const aops::PermuteOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUPermuteOpFactory final : public TypedOpFactory<OpTypes::kPermute, aops::PermuteOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::PermuteOpOptions& options) override {
    return std::make_shared<CPUPermuteOp>(options);
  }
};

}  // namespace mllm::cpu