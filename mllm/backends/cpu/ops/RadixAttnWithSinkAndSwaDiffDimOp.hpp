// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RadixAttnWithSinkAndSwaDiffDimOp.hpp"

namespace mllm::cpu {

class CPURadixAttnSwaSinkOp final : public aops::RadixAttnSwaSinkOp {
 public:
  explicit CPURadixAttnSwaSinkOp(const aops::RadixAttnSwaSinkOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURadixAttnSwaSinkOpFactory
    : public TypedOpFactory<OpTypes::kRadixAttnWithSinkAndSwaDiffDim, aops::RadixAttnSwaSinkOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RadixAttnSwaSinkOptions& options) override {
    return std::make_shared<CPURadixAttnSwaSinkOp>(options);
  }
};

}  // namespace mllm::cpu
