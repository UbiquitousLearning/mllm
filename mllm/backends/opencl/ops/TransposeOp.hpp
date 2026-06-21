// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/TransposeOp.hpp"

namespace mllm::opencl {

class OpenCLTransposeOp final : public aops::TransposeOp {
 public:
  explicit OpenCLTransposeOp(const aops::TransposeOpOptions& options);
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_generic_;
  std::shared_ptr<KernelWrap> kernel_0213_;
  std::shared_ptr<KernelWrap> kernel_0132_;
};

class OpenCLTransposeOpFactory : public TypedOpFactory<OpTypes::kTranspose, aops::TransposeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TransposeOpOptions& options) override {
    return std::make_shared<OpenCLTransposeOp>(options);
  }
};

}  // namespace mllm::opencl
