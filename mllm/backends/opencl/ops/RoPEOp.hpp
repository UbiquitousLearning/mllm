// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/RoPEOp.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

class OpenCLRoPEOp final : public aops::RoPEOp {
 public:
  explicit OpenCLRoPEOp(const aops::RoPEOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_f32_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_f16_ = nullptr;
};

class OpenCLRoPEOpFactory : public TypedOpFactory<OpTypes::kRoPE, aops::RoPEOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::RoPEOpOptions& options) override {
    return std::make_shared<OpenCLRoPEOp>(options);
  }
};

}  // namespace mllm::opencl
