// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/ReLUOp.hpp"

namespace mllm::cpu {

class CPUReLUOp final : public aops::ReLUOp {
 public:
  explicit CPUReLUOp(const aops::ReLUOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUReLUOpFactory : public TypedOpFactory<OpTypes::kReLU, aops::ReLUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ReLUOpOptions& options) override {
    return std::make_shared<CPUReLUOp>(options);
  }
};

}  // namespace mllm::cpu