// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"

namespace mllm::cpu {

class CPUCastTypeOp final : public aops::CastTypeOp {
 public:
  explicit CPUCastTypeOp(const aops::CastTypeOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUCastTypeOpFactory : public TypedOpFactory<OpTypes::kCastType, aops::CastTypeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CastTypeOpOptions& options) override {
    return std::make_shared<CPUCastTypeOp>(options);
  }
};

}  // namespace mllm::cpu
