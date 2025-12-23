// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

struct EqualOpOptions : public BaseOpOptions<EqualOpOptions> {};

class EqualOp : public BaseOp {
 public:
  explicit EqualOp(const EqualOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const EqualOpOptions& options() const { return options_; }

 protected:
  EqualOpOptions options_;
};

}  // namespace mllm::aops
