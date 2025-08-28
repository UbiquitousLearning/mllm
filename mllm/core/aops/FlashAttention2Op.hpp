// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct FlashAttention2OpOptions : public BaseOpOptions<FlashAttention2OpOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D;
  bool hp_exp = false;
  bool causal_mask = true;
};

class FlashAttention2Op : public BaseOp {
 public:
  explicit FlashAttention2Op(const FlashAttention2OpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const FlashAttention2OpOptions& options() const { return options_; }

 protected:
  FlashAttention2OpOptions options_;
};

}  // namespace mllm::aops
