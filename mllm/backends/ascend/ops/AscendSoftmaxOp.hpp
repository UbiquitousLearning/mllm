// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendSoftmaxOp final : public aops::SoftmaxOp {
 public:
  explicit AscendSoftmaxOp(const aops::SoftmaxOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendSoftmaxOpFactory final : public TypedOpFactory<OpTypes::kSoftmax, aops::SoftmaxOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SoftmaxOpOptions& options) override {
    return std::make_shared<AscendSoftmaxOp>(options);
  }
};

}  // namespace mllm::ascend