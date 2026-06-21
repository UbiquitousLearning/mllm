// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct TopKOpOptions : public BaseOpOptions<TopKOpOptions> {
  int k = 0;
  int dim = -1;
  bool largest = true;
  bool sorted = true;
};

class TopKOp : public BaseOp {
 public:
  explicit TopKOp(const TopKOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const TopKOpOptions& options() const { return options_; }

 protected:
  TopKOpOptions options_;
};

}  // namespace mllm::aops