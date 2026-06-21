// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SigmoidOp.hpp"

namespace mllm::cpu {

class CPUSigmoidOp final : public aops::SigmoidOp {
 public:
  explicit CPUSigmoidOp(const aops::SigmoidOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSigmoidOpFactory : public TypedOpFactory<OpTypes::kSigmoid, aops::SigmoidOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SigmoidOpOptions& options) override {
    return std::make_shared<CPUSigmoidOp>(options);
  }
};

}  // namespace mllm::cpu
