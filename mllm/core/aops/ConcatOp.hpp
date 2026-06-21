// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ConcatOpOptions : public BaseOpOptions<ConcatOpOptions> {
  int32_t dim;
};

class ConcatOp : public BaseOp {
 public:
  explicit ConcatOp(const ConcatOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const ConcatOpOptions& options() const { return options_; }

 protected:
  ConcatOpOptions options_;
};

}  // namespace mllm::aops
