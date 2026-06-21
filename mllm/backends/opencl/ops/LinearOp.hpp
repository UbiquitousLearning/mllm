// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::opencl {

class OpenCLLinearOp final : public aops::LinearOp {
 public:
  explicit OpenCLLinearOp(const aops::LinearOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_q4_0_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp32_q4_0_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_gemv_fp32_q4_0_transb_bias_ = nullptr;         // GEMV
  std::shared_ptr<KernelWrap> kernel_gemv_fp16_q4_0_transb_bias_ = nullptr;         // GEMV
  std::shared_ptr<KernelWrap> kernel_gemv_fp16_q4_0_transb_bias_half16_ = nullptr;  // GEMV for K%16==0
};

class OpenCLLinearOpFactory : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<OpenCLLinearOp>(options);
  }
};

}  // namespace mllm::opencl
