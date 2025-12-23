// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/WhereOp.hpp"

namespace mllm::cpu {

class CPUWhereOp final : public aops::WhereOp {
 public:
  explicit CPUWhereOp(const aops::WhereOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUWhereOpFactory : public TypedOpFactory<OpTypes::kWhere, aops::WhereOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::WhereOpOptions& options) override {
    return std::make_shared<CPUWhereOp>(options);
  }
};

}  // namespace mllm::cpu
