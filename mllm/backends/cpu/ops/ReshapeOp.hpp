// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ReshapeOp.hpp"

namespace mllm::cpu {

class CPUReshapeOp final : public aops::ReshapeOp {
 public:
  explicit CPUReshapeOp(const aops::ReshapeOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUReshapeOpFactory : public TypedOpFactory<OpTypes::kReshape, aops::ReshapeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ReshapeOpOptions& options) override {
    return std::make_shared<CPUReshapeOp>(options);
  }
};

}  // namespace mllm::cpu
