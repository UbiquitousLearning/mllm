// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ConvTranspose1DOp.hpp"

namespace mllm::cpu {

class CPUConvTranspose1DOp final : public aops::ConvTranspose1DOp {
 public:
  explicit CPUConvTranspose1DOp(const aops::ConvTranspose1DOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUConvTranspose1DOpFactory : public TypedOpFactory<OpTypes::kConvTranspose1D, aops::ConvTranspose1DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ConvTranspose1DOpOptions& options) override {
    return std::make_shared<CPUConvTranspose1DOp>(options);
  }
};

}  // namespace mllm::cpu
