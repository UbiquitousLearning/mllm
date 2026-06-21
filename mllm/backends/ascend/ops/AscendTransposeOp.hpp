// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendTransposeOp final : public aops::TransposeOp {
 public:
  explicit AscendTransposeOp(const aops::TransposeOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendTransposeOpFactory final : public TypedOpFactory<OpTypes::kTranspose, aops::TransposeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TransposeOpOptions& options) override {
    return std::make_shared<AscendTransposeOp>(options);
  }
};

}  // namespace mllm::ascend
