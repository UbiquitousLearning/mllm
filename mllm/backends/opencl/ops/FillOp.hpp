// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/FillOp.hpp"
#include <vector>

namespace mllm::opencl {
class OpenCLFillOp final : public aops::FillOp {
 public:
  explicit OpenCLFillOp(const aops::FillOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_arange_fp32_buffer_ = nullptr;
};

class OpenCLFillOpFactory : public TypedOpFactory<OpTypes::kFill, aops::FillOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FillOpOptions& options) override {
    return std::make_shared<OpenCLFillOp>(options);
  }
};

}  // namespace mllm::opencl
