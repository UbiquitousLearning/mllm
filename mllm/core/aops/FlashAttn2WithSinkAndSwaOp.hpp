// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct FlashAttention2SwaSinkOptions : public mllm::BaseOpOptions<FlashAttention2SwaSinkOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
  int32_t cur_seq_len;
  int sliding_window = -1;
  bool s_aux_enable = false;
};

class FlashAttention2SwaSinkOp : public BaseOp {
 public:
  explicit FlashAttention2SwaSinkOp(const FlashAttention2SwaSinkOptions& options);

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  FlashAttention2SwaSinkOptions options_;
};

}  // namespace mllm::aops
