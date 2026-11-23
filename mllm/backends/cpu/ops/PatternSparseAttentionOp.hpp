// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/PatternSparseAttentionOp.hpp"

namespace mllm::cpu::shadownpu {
class HMXInt8Selector;
}

namespace mllm::cpu {

class CPUPatternSparseAttentionOp final : public aops::PatternSparseAttentionOp {
 public:
  explicit CPUPatternSparseAttentionOp(const aops::PatternSparseAttentionOpOptions& options);

  ~CPUPatternSparseAttentionOp();

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  void prepareHmxRowPlan(int32_t query_heads, int32_t query_len, int32_t key_len);

  std::unique_ptr<shadownpu::HMXInt8Selector> hmx_selector_;
  std::vector<int32_t> hmx_row_offsets_;
  std::vector<const float*> hmx_query_heads_;
  std::vector<const uint16_t*> hmx_key_heads_;
  size_t hmx_topk_head_stride_ = 0;
};

class CPUPatternSparseAttentionOpFactory
    : public TypedOpFactory<OpTypes::kPatternSparseAttention, aops::PatternSparseAttentionOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::PatternSparseAttentionOpOptions& options) override {
    return std::make_shared<CPUPatternSparseAttentionOp>(options);
  }
};

}  // namespace mllm::cpu
