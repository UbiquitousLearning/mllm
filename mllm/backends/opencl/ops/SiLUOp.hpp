// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include <vector>

namespace mllm::opencl {

class OpenCLSiLUOp final : public aops::SiLUOp {
 public:
  explicit OpenCLSiLUOp(const aops::SiLUOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_buffer_ = nullptr;
};

class OpenCLSiLUOpFactory : public TypedOpFactory<OpTypes::kSiLU, aops::SiLUOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SiLUOpOptions& options) override {
    return std::make_shared<OpenCLSiLUOp>(options);
  }
};

}  // namespace mllm::opencl
