// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RoPEOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendRoPEOp final : public aops::RoPEOp {
 public:
  explicit AscendRoPEOp(const aops::RoPEOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendRoPEOpFactory final : public TypedOpFactory<OpTypes::kRoPE, aops::RoPEOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RoPEOpOptions& options) override {
    return std::make_shared<AscendRoPEOp>(options);
  }
};

}  // namespace mllm::ascend
