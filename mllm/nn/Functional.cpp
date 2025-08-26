// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Functional.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn::functional {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A, bool transpose_B, aops::MatMulOpType type) {
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kMatMul, aops::MatMulOpOptions{.transpose_a = transpose_A, .transpose_b = transpose_B, .matmul_type = type},
      {A, B})[0];
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

Tensor flashAttention2(const Tensor& Q, const Tensor& K, const Tensor& V) {
  // Inputs is all BSHD format.

  auto q_heads = Q.shape()[2];
  auto k_heads = K.shape()[2];
  auto B = Q.shape()[0];
  auto D = Q.shape()[3];

  return Context::instance().buildOpAndSubmitTask(OpTypes::kFlashAttention2,
                                                  aops::FlashAttention2OpOptions{
                                                      .B = B,
                                                      .q_head = q_heads,
                                                      .kv_head = k_heads,
                                                      .D = D,
                                                      .hp_exp = false,
                                                      .causal_mask = true,
                                                  },
                                                  {Q, K, V})[0];
}

Tensor softmax(const Tensor& x, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSoftmax, aops::SoftmaxOpOptions{.axis = dim}, {x})[0];
}

Tensor log(const Tensor& x) { return Context::instance().buildOpAndSubmitTask(OpTypes::kLog, aops::LogOpOptions{}, {x})[0]; }

std::array<Tensor, 2> topk(const Tensor& x, int32_t k, int32_t dim, bool largest, bool sorted) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kTopK, aops::TopKOpOptions{.k = k, .dim = dim, .largest = largest, .sorted = sorted}, {x});
  return {outputs[0], outputs[1]};
}

Tensor clip(const Tensor& x, float min_val, float max_val) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kClip, aops::ClipOpOptions{.min_val = min_val, .max_val = max_val},
                                                  {x})[0];
}

Tensor min(const Tensor& x, int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMin,
                                                  aops::ReduceMinOpOptions{.dim = dim, .keep_dim = keep_dim}, {x})[0];
}

Tensor max(const Tensor& x, int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMax,
                                                  aops::ReduceMaxOpOptions{.dim = dim, .keep_dim = keep_dim}, {x})[0];
}

Tensor sum(const Tensor& x, int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceSum,
                                                  aops::ReduceSumOpOptions{.dim = dim, .keep_dim = keep_dim}, {x})[0];
}

Tensor mean(const Tensor& x, int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMean, aops::MeanOpOptions{.dim = dim, .keep_dim = keep_dim},
                                                  {x})[0];
}

}  // namespace mllm::nn::functional
