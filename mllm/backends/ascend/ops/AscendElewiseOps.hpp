// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendAddOp final : public aops::AddOp {
 public:
  explicit AscendAddOp(const aops::AddOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendAddOpFactory final : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<AscendAddOp>(options);
  }
};

}  // namespace mllm::ascend