// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/PagedAttnOp.hpp"

namespace mllm::cpu {

class CPUPagedAttnOp final : public aops::PagedAttnOp {
 public:
  explicit CPUPagedAttnOp(const aops::PagedAttnOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUPagedAttnOpFactory : public TypedOpFactory<OpTypes::kPagedAttn, aops::PagedAttnOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::PagedAttnOpOptions& options) override {
    return std::make_shared<CPUPagedAttnOp>(options);
  }
};

}  // namespace mllm::cpu
