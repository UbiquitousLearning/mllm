// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendX2XOp final : public aops::X2XOp {
 public:
  explicit AscendX2XOp(const aops::X2XOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendX2XOpFactory final : public TypedOpFactory<OpTypes::kX2X, aops::X2XOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::X2XOpOptions& options) override {
    return std::make_shared<AscendX2XOp>(options);
  }
};

}  // namespace mllm::ascend

