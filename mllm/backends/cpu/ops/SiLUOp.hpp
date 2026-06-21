// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"

namespace mllm::cpu {

class CPUSiLUOp final : public aops::SiLUOp {
 public:
  explicit CPUSiLUOp(const aops::SiLUOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSiLUOpFactory : public TypedOpFactory<OpTypes::kSiLU, aops::SiLUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SiLUOpOptions& options) override {
    return std::make_shared<CPUSiLUOp>(options);
  }
};

}  // namespace mllm::cpu
