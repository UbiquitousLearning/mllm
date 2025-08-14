// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/MultimodalRoPEOp.hpp"

namespace mllm::cpu {

struct Qwen2VLMultimodalRoPEOpImpl {
  Tensor makeInvFreq(int output_dim, float rope_theta);

  std::pair<Tensor, Tensor> makePositionEmbedding(Tensor& position_ids, Tensor& inv_freq, int seq_len, int output_dim,
                                                  std::vector<int32_t>& mrope_section);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin, Tensor& cos);
};

class CPUMultimodalRoPEOp final : public aops::MultimodalRoPEOp {
 public:
  explicit CPUMultimodalRoPEOp(const aops::MultimodalRoPEOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUMultimodalRoPEOpFactory : public TypedOpFactory<OpTypes::kMultimodalRoPE, aops::MultimodalRoPEOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MultimodalRoPEOpOptions& options) override {
    return std::make_shared<CPUMultimodalRoPEOp>(options);
  }
};

}  // namespace mllm::cpu
