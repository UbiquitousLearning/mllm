// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Functional.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/core/aops/Scatter2ShardsOp.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/aops/PadOp.hpp"
#include "mllm/core/aops/MaskedScatterOp.hpp"
#include "mllm/core/aops/InterpolateOp.hpp"
#include "mllm/core/aops/StackOp.hpp"
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

Tensor stack(const std::vector<Tensor>& ins, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kStack, aops::StackOpOptions{.dim = dim}, ins)[0];
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

Tensor exp(const Tensor& x) { return Context::instance().buildOpAndSubmitTask(OpTypes::kExp, aops::ExpOpOptions{}, {x})[0]; }

Tensor sin(const Tensor& x) { return Context::instance().buildOpAndSubmitTask(OpTypes::kSin, aops::SinOpOptions{}, {x})[0]; }

Tensor cos(const Tensor& x) { return Context::instance().buildOpAndSubmitTask(OpTypes::kCos, aops::CosOpOptions{}, {x})[0]; }

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

Tensor silu(const Tensor& x) { return Context::instance().buildOpAndSubmitTask(OpTypes::kSiLU, aops::SiLUOpOptions{}, {x})[0]; }

Tensor silu_(const Tensor& x) {
  auto opt = aops::SiLUOpOptions{};
  opt.setInplace(true);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSiLU, opt, {x})[0];
}

void scatter2Shards(const Tensor& src, const Tensor& shards_pointer, int32_t dim) {
  Context::instance().buildOpAndSubmitTask(OpTypes::kScatter2Shards, aops::Scatter2ShardsOpOptions{.dim = dim},
                                           {src, shards_pointer});
}

Tensor scaledDotProductAttention(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask) {
  float scale = Q.size(-1);
  scale = (1.f / sqrtf((float)scale));
  auto attn_weight = matmul(Q, K, false, true) * scale;
  if (mask) { attn_weight = attn_weight + mask; }
  attn_weight = softmax(attn_weight, -1);
  return matmul(attn_weight, V);
}

Tensor pad(const Tensor& x, const std::vector<int32_t>& pad, aops::PadMode mode, float value) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kPad, aops::PadOpOptions{.pad = pad, .mode = mode, .value = value},
                                                  {x})[0];
}

Tensor interpolateBySize(const Tensor& x, const std::vector<int32_t>& size, aops::InterpolateOpMode mode, bool align_corners,
                         bool antialias) {
  aops::InterpolateOpOptions opts{};
  opts.size.assign(size.begin(), size.end());
  opts.mode = mode;
  opts.align_corners = align_corners;
  opts.antialias = antialias;
  return Context::instance().buildOpAndSubmitTask(OpTypes::kInterpolate, opts, {x})[0];
}

Tensor interpolateByScale(const Tensor& x, const std::vector<float>& scale_factor, aops::InterpolateOpMode mode,
                          bool align_corners, bool antialias) {
  aops::InterpolateOpOptions opts{};
  opts.scale_factor = scale_factor;
  opts.mode = mode;
  opts.align_corners = align_corners;
  opts.antialias = antialias;
  return Context::instance().buildOpAndSubmitTask(OpTypes::kInterpolate, opts, {x})[0];
}

void maskedScatter(const Tensor& dst, const Tensor& mask, const Tensor& src) {
  Context::instance().buildOpAndSubmitTask(OpTypes::kMaskedScatter, aops::MaskedScatterOpOptions{}, {dst, mask, src});
}

}  // namespace mllm::nn::functional
