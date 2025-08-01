// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Functional.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn::functional {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A, bool transpose_B) {
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kMatMul, aops::MatMulOpOptions{.transpose_a = transpose_A, .transpose_b = transpose_B}, {A, B})[0];
}

Tensor view(const Tensor& x, const std::vector<int32_t>& shape) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kView, aops::ViewOpOptions{.to_shape = shape}, {x})[0];
}

std::vector<Tensor> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {split_size_or_sections}}, {x});
}

std::vector<Tensor> split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = split_size_or_sections}, {x});
}

Tensor concat(const std::vector<Tensor>& ins, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kConcat, aops::ConcatOpOptions{.dim = dim}, ins)[0];
}

}  // namespace mllm::nn::functional
