// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"

namespace mllm::cpu {

class CPUEmbeddingOp final : public aops::EmbeddingOp {
 public:
  explicit CPUEmbeddingOp(const aops::EmbeddingOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUEmbeddingOpFactory : public TypedOpFactory<OpTypes::kEmbedding, aops::EmbeddingOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::EmbeddingOpOptions& options) override {
    return std::make_shared<CPUEmbeddingOp>(options);
  }
};

}  // namespace mllm::cpu
