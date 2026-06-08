// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CausalMaskOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendCausalMaskOp final : public aops::CausalMaskOp {
 public:
  explicit AscendCausalMaskOp(const aops::CausalMaskOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendCausalMaskOpFactory final : public TypedOpFactory<OpTypes::kCausalMask, aops::CausalMaskOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CausalMaskOpOptions& options) override {
    return std::make_shared<AscendCausalMaskOp>(options);
  }
};

}  // namespace mllm::ascend
