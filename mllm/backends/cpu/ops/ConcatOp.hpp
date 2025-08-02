// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ConcatOp.hpp"

namespace mllm::cpu {

class CPUConcatOp final : public aops::ConcatOp {
 public:
  explicit CPUConcatOp(const aops::ConcatOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUConcatOpFactory : public TypedOpFactory<OpTypes::kConcat, aops::ConcatOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ConcatOpOptions& options) override {
    return std::make_shared<CPUConcatOp>(options);
  }
};

}  // namespace mllm::cpu