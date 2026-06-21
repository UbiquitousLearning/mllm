// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"

namespace mllm::aops {

struct GatherOpOptions : public BaseOpOptions<GatherOpOptions> {
  int dim;
};

class GatherOp : public BaseOp {
 public:
  explicit GatherOp(const GatherOpOptions& options);

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GatherOpOptions& options() const { return options_; }

 protected:
  GatherOpOptions options_;
};

}  // namespace mllm::aops
