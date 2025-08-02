// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CloneOp.hpp"

namespace mllm::cpu {

class CPUCloneOp final : public aops::CloneOp {
 public:
  explicit CPUCloneOp(const aops::CloneOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUCloneOpFactory : public TypedOpFactory<OpTypes::kClone, aops::CloneOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CloneOpOptions& options) override {
    return std::make_shared<CPUCloneOp>(options);
  }
};

}  // namespace mllm::cpu
