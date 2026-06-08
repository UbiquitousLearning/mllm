// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

/// AscendEmbeddingOp implements Embedding using ACLNN aclnnEmbedding.
/// This is used instead of ATB GatherParam because ATB Gather is not supported on Ascend 310B.
class AscendEmbeddingOp final : public aops::EmbeddingOp {
 public:
  explicit AscendEmbeddingOp(const aops::EmbeddingOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendEmbeddingOpFactory final : public TypedOpFactory<OpTypes::kEmbedding, aops::EmbeddingOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::EmbeddingOpOptions& options) override {
    return std::make_shared<AscendEmbeddingOp>(options);
  }
};

}  // namespace mllm::ascend
