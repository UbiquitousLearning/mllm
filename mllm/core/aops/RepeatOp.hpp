// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct RepeatOpOptions : public BaseOpOptions<RepeatOpOptions> {
  int32_t dim;
  int32_t repeat_times;
};

class RepeatOp : public BaseOp {
 public:
  explicit RepeatOp(const RepeatOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const RepeatOpOptions& options() const { return options_; }

 protected:
  RepeatOpOptions options_;
};

}  // namespace mllm::aops