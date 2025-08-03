// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GELUOp.hpp"

namespace mllm::cpu {

class CPUGELUOp final : public aops::GELUOp {
 public:
  explicit CPUGELUOp(const aops::GELUOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUGELUOpFactory : public TypedOpFactory<OpTypes::kGELU, aops::GELUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GELUOpOptions& options) override {
    return std::make_shared<CPUGELUOp>(options);
  }
};

}  // namespace mllm::cpu
