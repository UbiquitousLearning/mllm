// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp" // 记得加上这个头文件
#include <vector>

namespace mllm::aops {

struct GDNOpOptions {
    int heads = 16;
    int head_dim = 128;
};

class GDNOp : public BaseOp {
 public:
  explicit GDNOp(const GDNOpOptions& options) : BaseOp(OpTypes::kGDN), options_(options) {}

  void load(const ParameterFile::ptr_t& ploader) override {}
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}

  inline const GDNOpOptions& options() const { return options_; }

 protected:
  GDNOpOptions options_;
};

} // namespace mllm::aops
