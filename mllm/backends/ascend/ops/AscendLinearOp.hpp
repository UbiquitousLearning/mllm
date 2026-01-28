// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendLinearOp final : public aops::LinearOp {
 public:
  explicit AscendLinearOp(const aops::LinearOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendLinearOpFactory final : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<AscendLinearOp>(options);
  }
};

}  // namespace mllm::ascend
