// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class PagedAttnImplType {
  kDefault = 0,
  kAllFp32 = 1,
};

struct PagedAttnOpOptions : public BaseOpOptions<PagedAttnOpOptions> {
  int32_t head_repeat_times = 1;
  bool high_precision_exp = false;
  bool fuse_rope = false;
  bool custom_causal_mask = false;
  PagedAttnImplType impl_type = PagedAttnImplType::kAllFp32;
};

class PagedAttnOp : public BaseOp {
 public:
  explicit PagedAttnOp(const PagedAttnOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline const PagedAttnOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  PagedAttnOpOptions options_;
};

}  // namespace mllm::aops
