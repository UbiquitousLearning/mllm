// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct SoftmaxOpOptions : public BaseOpOptions<SoftmaxOpOptions> {
  int axis;
};

class SoftmaxOp : public BaseOp {
 public:
  explicit SoftmaxOp(const SoftmaxOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const SoftmaxOpOptions& options() const { return options_; }

 protected:
  SoftmaxOpOptions options_;
};

}  // namespace mllm::aops