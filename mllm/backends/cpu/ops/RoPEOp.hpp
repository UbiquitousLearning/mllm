// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RoPEOp.hpp"

namespace mllm::cpu {

struct RoPEOpImpl {
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin, Tensor& cos,
               aops::RoPEOpOptionsInputType input_layout_type);
};

class CPURoPEOp final : public aops::RoPEOp {
 public:
  explicit CPURoPEOp(const aops::RoPEOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPURoPEOpFactory : public TypedOpFactory<OpTypes::kRoPE, aops::RoPEOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RoPEOpOptions& options) override {
    return std::make_shared<CPURoPEOp>(options);
  }
};

}  // namespace mllm::cpu
