// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ContiguousOp.hpp"

namespace mllm::cpu {

class CPUContiguousOp final : public aops::ContiguousOp {
 public:
  explicit CPUContiguousOp(const aops::ContiguousOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUContiguousOpFactory : public TypedOpFactory<OpTypes::kContiguous, aops::ContiguousOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ContiguousOpOptions& options) override {
    return std::make_shared<CPUContiguousOp>(options);
  }
};

}  // namespace mllm::cpu
