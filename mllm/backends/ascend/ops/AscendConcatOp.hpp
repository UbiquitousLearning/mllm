// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendConcatOp final : public aops::ConcatOp {
 public:
  explicit AscendConcatOp(const aops::ConcatOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendConcatOpFactory final : public TypedOpFactory<OpTypes::kConcat, aops::ConcatOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ConcatOpOptions& options) override {
    return std::make_shared<AscendConcatOp>(options);
  }
};

}  // namespace mllm::ascend
