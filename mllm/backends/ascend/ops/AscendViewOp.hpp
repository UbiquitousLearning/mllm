// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"

namespace mllm::ascend {

class AscendViewOp final : public aops::ViewOp {
 public:
  explicit AscendViewOp(const aops::ViewOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendViewOpFactory final : public TypedOpFactory<OpTypes::kView, aops::ViewOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ViewOpOptions& options) override {
    return std::make_shared<AscendViewOp>(options);
  }
};

}  // namespace mllm::ascend