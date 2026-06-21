// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/CausalMaskOp.hpp"

namespace mllm::opencl {

class OpenCLCausalMaskOp final : public aops::CausalMaskOp {
 public:
  explicit OpenCLCausalMaskOp(const aops::CausalMaskOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_ = nullptr;
};

class OpenCLCausalMaskOpFactory : public TypedOpFactory<OpTypes::kCausalMask, aops::CausalMaskOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CausalMaskOpOptions& options) override {
    return std::make_shared<OpenCLCausalMaskOp>(options);
  }
};

}  // namespace mllm::opencl
