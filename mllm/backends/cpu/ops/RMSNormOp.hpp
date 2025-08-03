// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"

namespace mllm::cpu {

class CPURMSNormOp final : public aops::RMSNormOp {
 public:
  explicit CPURMSNormOp(const aops::RMSNormOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURMSNormOpFactory : public TypedOpFactory<OpTypes::kRMSNorm, aops::RMSNormOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RMSNormOpOptions& options) override {
    return std::make_shared<CPURMSNormOp>(options);
  }
};

}  // namespace mllm::cpu
