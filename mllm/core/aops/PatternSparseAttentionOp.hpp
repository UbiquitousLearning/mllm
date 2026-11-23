// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct PatternSparseAttentionOpOptions : public BaseOpOptions<PatternSparseAttentionOpOptions> {
  float sparsity = 0.0F;
  bool causal_mask = true;
  float local_ratio = 0.5F;
  int32_t prefix_tokens = 0;
  int32_t dense_tokens = 0;
  bool random_pattern = false;
  uint32_t random_seed = 1;
  int32_t pack_reserve_tokens = 0;
  bool hmx_selector = false;
  std::vector<float> head_retentions;
  int32_t layer_id = -1;
};

// Sparse attention over BHSD query, key and value tensors. HMX selection uses
// a fourth F16 selector-key input while attention math keeps the canonical key
// dtype from the second input. The selected-key set is deterministic for a
// fixed option set and tensor geometry.
class PatternSparseAttentionOp : public BaseOp {
 public:
  explicit PatternSparseAttentionOp(const PatternSparseAttentionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const PatternSparseAttentionOpOptions& options() const { return options_; }

 protected:
  PatternSparseAttentionOpOptions options_;
};

}  // namespace mllm::aops
