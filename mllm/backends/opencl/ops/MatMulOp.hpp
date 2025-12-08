// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

class OpenCLMatMulOp final : public aops::MatMulOp {
 public:
  explicit OpenCLMatMulOp(const aops::MatMulOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLMatMulOpFactory : public TypedOpFactory<OpTypes::kMatMul, aops::MatMulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MatMulOpOptions& options) override {
    return std::make_shared<OpenCLMatMulOp>(options);
  }
};

}  // namespace mllm::opencl
