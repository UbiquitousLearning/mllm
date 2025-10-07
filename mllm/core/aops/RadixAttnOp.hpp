// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct RadixAttnOpOptions : public BaseOpOptions<RadixAttnOpOptions> {
  int32_t H_Q;
  int32_t H_KV;
};

class RadixAttnOp : public BaseOp {
 public:
  explicit RadixAttnOp(const RadixAttnOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  RadixAttnOpOptions options_;
};

}  // namespace mllm::aops
