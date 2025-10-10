// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct Scatter2ShardsOpOptions : public BaseOpOptions<Scatter2ShardsOpOptions> {
  int dim = 0;
};

class Scatter2ShardsOp : public BaseOp {
 public:
  explicit Scatter2ShardsOp(const Scatter2ShardsOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  Scatter2ShardsOpOptions options_;
};

}  // namespace mllm::aops
