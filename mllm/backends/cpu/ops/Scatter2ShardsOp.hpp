// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Scatter2ShardsOp.hpp"

namespace mllm::cpu {

class CPUScatter2ShardsOp final : public aops::Scatter2ShardsOp {
 public:
  explicit CPUScatter2ShardsOp(const aops::Scatter2ShardsOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUScatter2ShardsOpFactory : public TypedOpFactory<OpTypes::kScatter2Shards, aops::Scatter2ShardsOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::Scatter2ShardsOpOptions& options) override {
    return std::make_shared<CPUScatter2ShardsOp>(options);
  }
};

}  // namespace mllm::cpu
