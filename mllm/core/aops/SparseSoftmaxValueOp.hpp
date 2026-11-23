// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <vector>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct SparseSoftmaxValueOpOptions : public BaseOpOptions<SparseSoftmaxValueOpOptions> {
  float sparsity = 0.0F;
  bool causal_mask = true;
  int32_t dense_tokens = 0;
  std::vector<float> head_retentions;
};

// Exact stable Top-k selection over F32 attention logits followed by softmax
// and multiplication with an F32/F16 BHSD value cache.
class SparseSoftmaxValueOp : public BaseOp {
 public:
  explicit SparseSoftmaxValueOp(const SparseSoftmaxValueOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const SparseSoftmaxValueOpOptions& options() const { return options_; }

 protected:
  SparseSoftmaxValueOpOptions options_;
};

}  // namespace mllm::aops
