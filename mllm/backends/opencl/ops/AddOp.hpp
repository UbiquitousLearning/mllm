// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include <vector>

namespace mllm::opencl {
class OpenCLAddOp final : public aops::AddOp {
 public:
  explicit OpenCLAddOp(const aops::AddOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLAddOpFactory : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<OpenCLAddOp>(options);
  }
};

}  // namespace mllm::opencl