// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendSliceOp final : public aops::SliceOp {
 public:
  explicit AscendSliceOp(const aops::SliceOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendSliceOpFactory final : public TypedOpFactory<OpTypes::kSlice, aops::SliceOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SliceOpOptions& options) override {
    return std::make_shared<AscendSliceOp>(options);
  }
};

}  // namespace mllm::ascend
