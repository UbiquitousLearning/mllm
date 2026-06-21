// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/STFTOp.hpp"

namespace mllm::cpu {

class CPUSTFTOp final : public aops::STFTOp {
 public:
  explicit CPUSTFTOp(const aops::STFTOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSTFTOpFactory : public TypedOpFactory<OpTypes::kSTFT, aops::STFTOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::STFTOpOptions& options) override {
    return std::make_shared<CPUSTFTOp>(options);
  }
};

}  // namespace mllm::cpu
