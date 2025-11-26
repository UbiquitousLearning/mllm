// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RadixAttnDiffDimOp.hpp"

namespace mllm::cpu {

class CPURadixAttnRelaxOp final : public aops::RadixAttnRelaxOp {
 public:
  explicit CPURadixAttnRelaxOp(const aops::RadixAttnRelaxOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURadixAttnRelaxOpFactory : public TypedOpFactory<OpTypes::kRadixAttnRelax, aops::RadixAttnRelaxOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RadixAttnRelaxOpOptions& options) override {
    return std::make_shared<CPURadixAttnRelaxOp>(options);
  }
};

}  // namespace mllm::cpu
