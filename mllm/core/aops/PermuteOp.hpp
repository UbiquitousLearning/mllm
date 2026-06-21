// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct PermuteOpOptions : public BaseOpOptions<PermuteOpOptions> {
  std::vector<int> axis;
};

class PermuteOp : public BaseOp {
 public:
  explicit PermuteOp(const PermuteOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const PermuteOpOptions& options() const { return options_; }

 protected:
  PermuteOpOptions options_;
};

}  // namespace mllm::aops