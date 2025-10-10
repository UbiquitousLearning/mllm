// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RadixAttnOp.hpp"

namespace mllm::cpu {

class CPURadixAttnOp final : public aops::RadixAttnOp {
 public:
  explicit CPURadixAttnOp(const aops::RadixAttnOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURadixAttnOpFactory : public TypedOpFactory<OpTypes::kRadixAttn, aops::RadixAttnOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RadixAttnOpOptions& options) override {
    return std::make_shared<CPURadixAttnOp>(options);
  }
};

}  // namespace mllm::cpu
