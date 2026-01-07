// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GatherOp.hpp"

namespace mllm::cpu {

class CPUGatherOp final : public aops::GatherOp {
 public:
  explicit CPUGatherOp(const aops::GatherOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUGatherOpFactory : public TypedOpFactory<OpTypes::kGather, aops::GatherOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GatherOpOptions& options) override {
    return std::make_shared<CPUGatherOp>(options);
  }
};

}  // namespace mllm::cpu
