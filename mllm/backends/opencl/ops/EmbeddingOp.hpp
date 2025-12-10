// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"

namespace mllm::opencl {

class OpenCLEmbeddingOp final : public aops::EmbeddingOp {
 public:
  explicit OpenCLEmbeddingOp(const aops::EmbeddingOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_q40_buffer_ = nullptr;
};

class OpenCLEmbeddingOpFactory : public TypedOpFactory<OpTypes::kEmbedding, aops::EmbeddingOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::EmbeddingOpOptions& options) override {
    return std::make_shared<OpenCLEmbeddingOp>(options);
  }
};

}  // namespace mllm::opencl
