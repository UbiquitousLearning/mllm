// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/KimiDeltaAttentionOp.hpp"

namespace mllm::cpu {

class CPUKimiDeltaAttentionOp final : public aops::KimiDeltaAttentionOp {
 public:
  explicit CPUKimiDeltaAttentionOp(const aops::KimiDeltaAttentionOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUKimiDeltaAttentionOpFactory : public TypedOpFactory<OpTypes::kKimiDeltaAttention, aops::KimiDeltaAttentionOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::KimiDeltaAttentionOpOptions& options) override {
    return std::make_shared<CPUKimiDeltaAttentionOp>(options);
  }
};

}  // namespace mllm::cpu
