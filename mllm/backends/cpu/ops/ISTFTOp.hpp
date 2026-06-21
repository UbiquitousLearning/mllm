// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ISTFTOp.hpp"

namespace mllm::cpu {

class CPUISTFTOp final : public aops::ISTFTOp {
 public:
  explicit CPUISTFTOp(const aops::ISTFTOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUISTFTOpFactory : public TypedOpFactory<OpTypes::kISTFT, aops::ISTFTOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ISTFTOpOptions& options) override {
    return std::make_shared<CPUISTFTOp>(options);
  }
};

// ====================================
// NOTE: implementation is in STFTOp.cpp
// ====================================

}  // namespace mllm::cpu