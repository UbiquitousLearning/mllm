// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Conv3DOp.hpp"

namespace mllm::cpu {

class CPUConv3DOp final : public aops::Conv3DOp {
 public:
  explicit CPUConv3DOp(const aops::Conv3DOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUConv3DOpFactory : public TypedOpFactory<OpTypes::kConv3D, aops::Conv3DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::Conv3DOpOptions& options) override {
    return std::make_shared<CPUConv3DOp>(options);
  }
};

}  // namespace mllm::cpu
