// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendSiLUOp final : public aops::SiLUOp {
 public:
  explicit AscendSiLUOp(const aops::SiLUOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendSiLUOpFactory final : public TypedOpFactory<OpTypes::kSiLU, aops::SiLUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SiLUOpOptions& options) override {
    return std::make_shared<AscendSiLUOp>(options);
  }
};

}  // namespace mllm::ascend
