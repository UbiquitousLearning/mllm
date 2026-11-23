// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class FlashAttention2Layout : int32_t { kBSHD = 0, kBHSD = 1 };

struct FlashAttention2OpOptions : public BaseOpOptions<FlashAttention2OpOptions> {
  int32_t B = 0;
  int32_t q_head = 0;
  int32_t kv_head = 0;
  int32_t D = 0;
  bool hp_exp = false;
  bool causal_mask = true;
  FlashAttention2Layout layout = FlashAttention2Layout::kBSHD;
  int32_t sliding_window = 0;
  // The tiled variant is algebraically equivalent but has a different
  // floating-point reduction order from the original per-key implementation.
  bool use_tiled = false;
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
