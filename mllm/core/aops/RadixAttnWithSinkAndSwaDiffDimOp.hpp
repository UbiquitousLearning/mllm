// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class RadixAttnSwaSinkPattern : uint8_t {
  kPrefill = 0,
  kDecode = 1,
  kAppend = 2,
};

struct RadixAttnSwaSinkOptions : public mllm::BaseOpOptions<RadixAttnSwaSinkOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
  int32_t cur_seq_len;
  int sliding_window = -1;
  bool s_aux_enable = false;
  RadixAttnSwaSinkPattern pattern = RadixAttnSwaSinkPattern::kDecode;
};

class RadixAttnSwaSinkOp : public BaseOp {
 public:
  explicit RadixAttnSwaSinkOp(const RadixAttnSwaSinkOptions& options);

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  RadixAttnSwaSinkOptions options_;
};

}  // namespace mllm::aops
