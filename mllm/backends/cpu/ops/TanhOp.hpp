// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/TanhOp.hpp"

namespace mllm::cpu {

class CPUTanhOp final : public aops::TanhOp {
 public:
  explicit CPUTanhOp(const aops::TanhOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUTanhOpFactory : public TypedOpFactory<OpTypes::kTanh, aops::TanhOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TanhOpOptions& options) override {
    return std::make_shared<CPUTanhOp>(options);
  }
};

}  // namespace mllm::cpu
