// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"
#include <vector>

namespace mllm::qnn {

class QNNEmbeddingOp final : public aops::EmbeddingOp {
 public:
  explicit QNNEmbeddingOp(const aops::EmbeddingOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QNNEmbeddingOpFactory : public TypedOpFactory<OpTypes::kEmbedding, aops::EmbeddingOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::EmbeddingOpOptions& options) override {
    return std::make_shared<QNNEmbeddingOp>(options);
  }
};

}  // namespace mllm::qnn