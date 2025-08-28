// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct QuickGELUOpOptions : public BaseOpOptions<QuickGELUOpOptions> {};

class QuickGELUOp : public BaseOp {
 public:
  explicit QuickGELUOp(const QuickGELUOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const QuickGELUOpOptions& options() const { return options_; }

 protected:
  QuickGELUOpOptions options_;
};

}  // namespace mllm::aops