// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include <vector>

namespace mllm::opencl {

class OpenCLX2XOp final : public aops::X2XOp {
 public:
  explicit OpenCLX2XOp(const aops::X2XOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class OpenCLX2XOpFactory : public TypedOpFactory<OpTypes::kX2X, aops::X2XOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::X2XOpOptions& options) override {
    return std::make_shared<OpenCLX2XOp>(options);
  }
};

}  // namespace mllm::opencl