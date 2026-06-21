// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class MatMulOpType {
  kDefault = 0,
  kGGUF = 1,

  // BLAS
  kBLAS = 2,

  // MLLM Self Host
  kMllmBlas,
};

struct MatMulOpOptions : public BaseOpOptions<MatMulOpOptions> {
  bool transpose_a = false;
  bool transpose_b = false;
  MatMulOpType matmul_type = MatMulOpType::kDefault;
};

inline MatMulOpType str2MatMulOpType(const std::string& str) {
  static const std::unordered_map<std::string, MatMulOpType> map = {{"Default", MatMulOpType::kDefault},
                                                                    {"GGUF", MatMulOpType::kGGUF},
                                                                    {"BLAS", MatMulOpType::kBLAS},
                                                                    {"MllmBlas", MatMulOpType::kMllmBlas}};

  auto it = map.find(str);
  if (it != map.end()) return it->second;
  return MatMulOpType::kDefault;
}

inline std::string MatMulOpType2Str(MatMulOpType type) {
  static const std::unordered_map<MatMulOpType, std::string> map = {{MatMulOpType::kDefault, "Default"},
                                                                    {MatMulOpType::kGGUF, "GGUF"},
                                                                    {MatMulOpType::kBLAS, "BLAS"},
                                                                    {MatMulOpType::kMllmBlas, "MllmBlas"}};

  auto it = map.find(type);
  if (it != map.end()) return it->second;
  return "Default";
}

class MatMulOp : public BaseOp {
 public:
  explicit MatMulOp(const MatMulOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const MatMulOpOptions& options() const { return options_; }

 protected:
  MatMulOpOptions options_;
};

}  // namespace mllm::aops
