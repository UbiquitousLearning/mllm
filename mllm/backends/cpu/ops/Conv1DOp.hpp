// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Conv1DOp.hpp"

namespace mllm::cpu {

class CPUConv1DOp final : public aops::Conv1DOp {
 public:
  explicit CPUConv1DOp(const aops::Conv1DOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUConv1DOpFactory : public TypedOpFactory<OpTypes::kConv1D, aops::Conv1DOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::Conv1DOpOptions& options) override {
    return std::make_shared<CPUConv1DOp>(options);
  }
};

}  // namespace mllm::cpu