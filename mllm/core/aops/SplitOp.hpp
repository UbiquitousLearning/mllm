// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct SplitOpOptions : public BaseOpOptions<SplitOpOptions> {
  int32_t dim;

  // if split_size_or_sections_.size() is 1, use split size. else split to sections.
  std::vector<int32_t> split_size_or_sections;
};

class SplitOp : public BaseOp {
 public:
  explicit SplitOp(const SplitOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const SplitOpOptions& options() const { return options_; }

 protected:
  SplitOpOptions options_;
};

}  // namespace mllm::aops