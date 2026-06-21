// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/VisionRoPEOp.hpp"

namespace mllm::cpu {

struct Qwen2VLVisionRoPEOpImpl {
  void forward(const Tensor& activation, const Tensor& sin, const Tensor& cos, Tensor& out);
};

class CPUVisionRoPEOp final : public aops::VisionRoPEOp {
 public:
  explicit CPUVisionRoPEOp(const aops::VisionRoPEOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUVisionRoPEOpFactory : public TypedOpFactory<OpTypes::kVisionRoPE, aops::VisionRoPEOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::VisionRoPEOpOptions& options) override {
    return std::make_shared<CPUVisionRoPEOp>(options);
  }
};

}  // namespace mllm::cpu
