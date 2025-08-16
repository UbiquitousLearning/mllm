// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RepeatOp.hpp"

namespace mllm::cpu {

class CPURepeatOp final : public aops::RepeatOp {
 public:
  explicit CPURepeatOp(const aops::RepeatOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURepeatOpFactory : public TypedOpFactory<OpTypes::kRepeat, aops::RepeatOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RepeatOpOptions& options) override {
    return std::make_shared<CPURepeatOp>(options);
  }
};

}  // namespace mllm::cpu
