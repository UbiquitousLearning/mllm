// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"

namespace mllm::cpu {

class CPUSoftmaxOp final : public aops::SoftmaxOp {
 public:
  explicit CPUSoftmaxOp(const aops::SoftmaxOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSoftmaxOpFactory : public TypedOpFactory<OpTypes::kSoftmax, aops::SoftmaxOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SoftmaxOpOptions& options) override {
    return std::make_shared<CPUSoftmaxOp>(options);
  }
};

}  // namespace mllm::cpu
