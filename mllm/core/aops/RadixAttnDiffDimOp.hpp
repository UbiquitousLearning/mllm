// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct RadixAttnRelaxOpOptions : public mllm::BaseOpOptions<RadixAttnRelaxOpOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
};

class RadixAttnRelaxOp : public BaseOp {
 public:
  explicit RadixAttnRelaxOp(const RadixAttnRelaxOpOptions& options);

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  RadixAttnRelaxOpOptions options_;
};

}  // namespace mllm::aops
