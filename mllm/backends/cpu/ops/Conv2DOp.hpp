// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Conv2DOp.hpp"

namespace mllm::cpu {

class CPUConv2DOp final : public aops::Conv2DOp {
 public:
  explicit CPUConv2DOp(const aops::Conv2DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUConv2DOpFactory : public TypedOpFactory<OpTypes::kConv2D, aops::Conv2DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::Conv2DOpOptions& options) override {
    return std::make_shared<CPUConv2DOp>(options);
  }
};

}  // namespace mllm::cpu
