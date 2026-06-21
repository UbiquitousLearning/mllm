// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

class OpenCLRMSNormOp final : public aops::RMSNormOp {
 public:
  explicit OpenCLRMSNormOp(const aops::RMSNormOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_f32_q4_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_f16_q4_ = nullptr;
};

class OpenCLRMSNormOpFactory : public TypedOpFactory<OpTypes::kRMSNorm, aops::RMSNormOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RMSNormOpOptions& options) override {
    return std::make_shared<OpenCLRMSNormOp>(options);
  }
};

}  // namespace mllm::opencl
