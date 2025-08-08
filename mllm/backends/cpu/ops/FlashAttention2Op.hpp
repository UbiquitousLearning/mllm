// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"

namespace mllm::cpu {

class CPUFlashAttention2Op final : public aops::FlashAttention2Op {
 public:
  explicit CPUFlashAttention2Op(const aops::FlashAttention2OpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUFlashAttention2OpFactory : public TypedOpFactory<OpTypes::kFlashAttention2, aops::FlashAttention2OpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FlashAttention2OpOptions& options) override {
    return std::make_shared<CPUFlashAttention2Op>(options);
  }
};

}  // namespace mllm::cpu
