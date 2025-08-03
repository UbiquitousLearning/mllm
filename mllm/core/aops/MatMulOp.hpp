// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class MatMulOpType {
  kDefault = 0,
  kLlamaFile = 1,
};

struct MatMulOpOptions : public BaseOpOptions<MatMulOpOptions> {
  bool transpose_a = false;
  bool transpose_b = false;
  MatMulOpType matmul_type = MatMulOpType::kDefault;
};

class MatMulOp : public BaseOp {
 public:
  explicit MatMulOp(const MatMulOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  MatMulOpOptions options_;
};

}  // namespace mllm::aops