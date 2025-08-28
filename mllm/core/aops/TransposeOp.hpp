// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct TransposeOpOptions : public BaseOpOptions<TransposeOpOptions> {
  int dim0;
  int dim1;
};

class TransposeOp : public BaseOp {
 public:
  explicit TransposeOp(const TransposeOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const TransposeOpOptions& options() const { return options_; }

 protected:
  TransposeOpOptions options_;
};

}  // namespace mllm::aops