// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ArgsortOpOptions : public BaseOpOptions<ArgsortOpOptions> {
  int dim = -1;
  bool descending = false;
};

class ArgsortOp : public BaseOp {
 public:
  explicit ArgsortOp(const ArgsortOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline ArgsortOpOptions& options() { return options_; }

 protected:
  ArgsortOpOptions options_;
};

}  // namespace mllm::aops
