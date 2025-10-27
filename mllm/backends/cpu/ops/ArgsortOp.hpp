// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/ArgsortOp.hpp"

namespace mllm::cpu {

class CPUArgsortOp final : public aops::ArgsortOp {
 public:
  explicit CPUArgsortOp(const aops::ArgsortOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUArgsortOpFactory final : public TypedOpFactory<OpTypes::kArgsort, aops::ArgsortOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ArgsortOpOptions& options) override {
    return std::make_shared<CPUArgsortOp>(options);
  }
};

}  // namespace mllm::cpu