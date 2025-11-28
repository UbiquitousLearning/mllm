// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/FlashAttn2WithSinkAndSwaOp.hpp"

namespace mllm::cpu {

class CPUFlashAttention2SwaSinkOp final : public aops::FlashAttention2SwaSinkOp {
 public:
  explicit CPUFlashAttention2SwaSinkOp(const aops::FlashAttention2SwaSinkOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUFlashAttention2SwaSinkOpFactory
    : public TypedOpFactory<OpTypes::kFlashAttention2WithSinkAndSwa, aops::FlashAttention2SwaSinkOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FlashAttention2SwaSinkOptions& options) override {
    return std::make_shared<CPUFlashAttention2SwaSinkOp>(options);
  }
};

}  // namespace mllm::cpu
