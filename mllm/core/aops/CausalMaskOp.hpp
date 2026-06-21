// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct CausalMaskOpOptions : public BaseOpOptions<CausalMaskOpOptions> {
  bool sliding_window = false;
  int32_t window_size = 0;
};

class CausalMaskOp : public BaseOp {
 public:
  explicit CausalMaskOp(const CausalMaskOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const CausalMaskOpOptions& options() const { return options_; }

 protected:
  CausalMaskOpOptions options_;
};

}  // namespace mllm::aops
