// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"

namespace mllm::opencl {

class OpenCLSoftmaxOp final : public aops::SoftmaxOp {
 public:
  explicit OpenCLSoftmaxOp(const aops::SoftmaxOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_ = nullptr;
};

class OpenCLSoftmaxOpFactory : public TypedOpFactory<OpTypes::kSoftmax, aops::SoftmaxOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SoftmaxOpOptions& options) override {
    return std::make_shared<OpenCLSoftmaxOp>(options);
  }
};

}  // namespace mllm::opencl
