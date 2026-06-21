// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendFillOp final : public aops::FillOp {
 public:
  explicit AscendFillOp(const aops::FillOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendFillOpFactory final : public TypedOpFactory<OpTypes::kFill, aops::FillOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FillOpOptions& options) override {
    return std::make_shared<AscendFillOp>(options);
  }
};

}  // namespace mllm::ascend
