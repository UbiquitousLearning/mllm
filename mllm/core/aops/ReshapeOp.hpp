// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ReshapeOpOptions : public BaseOpOptions<ReshapeOpOptions> {
  std::vector<int32_t> shape;
};

class ReshapeOp : public BaseOp {
 public:
  explicit ReshapeOp(const ReshapeOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const ReshapeOpOptions& options() const { return options_; }

 protected:
  ReshapeOpOptions options_;
};

}  // namespace mllm::aops