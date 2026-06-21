// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/PadOp.hpp"

namespace mllm::cpu {

class CPUPadOp final : public aops::PadOp {
 public:
  explicit CPUPadOp(const aops::PadOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUPadOpFactory : public TypedOpFactory<OpTypes::kPad, aops::PadOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::PadOpOptions& options) override {
    return std::make_shared<CPUPadOp>(options);
  }
};

}  // namespace mllm::cpu
