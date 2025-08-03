// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SplitOp.hpp"

namespace mllm::cpu {

class CPUSplitOp final : public aops::SplitOp {
 public:
  explicit CPUSplitOp(const aops::SplitOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSplitOpFactory : public TypedOpFactory<OpTypes::kSplit, aops::SplitOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SplitOpOptions& options) override {
    return std::make_shared<CPUSplitOp>(options);
  }
};

}  // namespace mllm::cpu
