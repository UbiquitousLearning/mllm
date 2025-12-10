// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CopyOp.hpp"

namespace mllm::opencl {

class OpenCLCopyOp final : public aops::CopyOp {
 public:
  explicit OpenCLCopyOp(const aops::CopyOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class OpenCLCopyOpFactory : public TypedOpFactory<OpTypes::kCopy, aops::CopyOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CopyOpOptions& options) override {
    return std::make_shared<OpenCLCopyOp>(options);
  }
};

}  // namespace mllm::opencl
