// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/AvgPool1dOp.hpp"

namespace mllm::cpu {

class CPUAvgPool1dOp final : public aops::AvgPool1dOp {
 public:
  explicit CPUAvgPool1dOp(const aops::AvgPool1dOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUAvgPool1dOpFactory : public TypedOpFactory<OpTypes::kAvgPool1d, aops::AvgPool1dOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AvgPool1dOpOptions& options) override {
    return std::make_shared<CPUAvgPool1dOp>(options);
  }
};

}  // namespace mllm::cpu
