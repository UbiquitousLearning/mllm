// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/InterpolateOp.hpp"

namespace mllm::cpu {

class CPUInterpolateOp final : public aops::InterpolateOp {
 public:
  explicit CPUInterpolateOp(const aops::InterpolateOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUInterpolateOpFactory : public TypedOpFactory<OpTypes::kInterpolate, aops::InterpolateOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::InterpolateOpOptions& options) override {
    return std::make_shared<CPUInterpolateOp>(options);
  }
};

}  // namespace mllm::cpu
