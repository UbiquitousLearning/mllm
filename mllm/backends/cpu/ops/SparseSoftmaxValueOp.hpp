// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SparseSoftmaxValueOp.hpp"

namespace mllm::cpu {

class CPUSparseSoftmaxValueOp final : public aops::SparseSoftmaxValueOp {
 public:
  explicit CPUSparseSoftmaxValueOp(const aops::SparseSoftmaxValueOpOptions& options) : aops::SparseSoftmaxValueOp(options) {}

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSparseSoftmaxValueOpFactory : public TypedOpFactory<OpTypes::kSparseSoftmaxValue, aops::SparseSoftmaxValueOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SparseSoftmaxValueOpOptions& options) override {
    return std::make_shared<CPUSparseSoftmaxValueOp>(options);
  }
};

}  // namespace mllm::cpu
