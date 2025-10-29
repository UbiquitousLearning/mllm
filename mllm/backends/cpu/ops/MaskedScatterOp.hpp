// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/MaskedScatterOp.hpp"

namespace mllm::cpu {

class CPUMaskedScatterOp final : public aops::MaskedScatterOp {
 public:
  explicit CPUMaskedScatterOp(const aops::MaskedScatterOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUMaskedScatterOpFactory : public TypedOpFactory<OpTypes::kMaskedScatter, aops::MaskedScatterOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MaskedScatterOpOptions& options) override {
    return std::make_shared<CPUMaskedScatterOp>(options);
  }
};

}  // namespace mllm::cpu
