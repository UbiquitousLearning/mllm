// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/TopKOp.hpp"

namespace mllm::cpu {

class CPUTopKOp final : public aops::TopKOp {
 public:
  explicit CPUTopKOp(const aops::TopKOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUTopKOpFactory final : public TypedOpFactory<OpTypes::kTopK, aops::TopKOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TopKOpOptions& options) override {
    return std::make_shared<CPUTopKOp>(options);
  }
};

}  // namespace mllm::cpu
