// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <stdexcept>
#include "mllm/backends/cpu/ops/FlashAttention2Op.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/fwd_bshd.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/fwd_tiled.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

namespace {

bool hasDenseStrides(const Tensor& tensor) {
  const auto shape = tensor.shape(), strides = tensor.stride();
  int64_t expected = 1;
  for (int i = 3; i >= 0; --i) {
    if (shape[i] > 1 && strides[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

std::array<int64_t, 4> canonicalStrides(const Tensor& tensor, int h, int s) {
  const auto strides = tensor.stride();
  return {strides[0], strides[h], strides[s], strides[3]};
}

}  // namespace

CPUFlashAttention2Op::CPUFlashAttention2Op(const aops::FlashAttention2OpOptions& options) : aops::FlashAttention2Op(options) {}

void CPUFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& O = outputs[0];

  if (Q.dtype() != kFloat32) throw std::invalid_argument("CPU FlashAttention2 requires FP32 Q, K and V");
  const bool sequence_first = options_.layout == aops::FlashAttention2Layout::kBSHD;
  const int h = sequence_first ? 2 : 1;
  const int s = 3 - h;
  const auto q = Q.shape(), k = K.shape(), v = V.shape();
  const flash_attn2::TiledAttentionShape shape{
      q[0], q[h], k[h], q[s], k[s], q[3], v[3], options_.causal_mask, options_.sliding_window};
  const int threads = options_.getThreads();
  const bool legacy_geometry = sequence_first && shape.causal && shape.sliding_window == 0 && q[3] == v[3] && hasDenseStrides(Q)
                               && hasDenseStrides(K)
                               && hasDenseStrides(V)
                               // The legacy pointer arithmetic uses int32_t offsets.
                               && Q.numel() <= INT32_MAX && K.numel() <= INT32_MAX && V.numel() <= INT32_MAX;
  if (!options_.use_tiled && legacy_geometry) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
    flash_attn2::fwd_bshd<flash_attn2::details::__ArmArchTag, float, float, float, float, float, true, true>(
        shape.batch, shape.query_heads, shape.kv_heads, shape.query_tokens, shape.kv_tokens, shape.query_dim, Q.ptr<float>(),
        K.ptr<float>(), V.ptr<float>(), O.ptr<float>(), threads);
    return;
#elif defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)
    flash_attn2::fwd_bshd<flash_attn2::details::__X86ArchTag, float, float, float, float, float>(
        shape.batch, shape.query_heads, shape.kv_heads, shape.query_tokens, shape.kv_tokens, shape.query_dim, Q.ptr<float>(),
        K.ptr<float>(), V.ptr<float>(), O.ptr<float>(), threads);
    return;
#endif
  }
  flash_attn2::fwdTiledFp32(shape, {Q.ptr<float>(), canonicalStrides(Q, h, s)}, {K.ptr<float>(), canonicalStrides(K, h, s)},
                            {V.ptr<float>(), canonicalStrides(V, h, s)}, {O.ptr<float>(), canonicalStrides(O, h, s)}, threads);
}

}  // namespace mllm::cpu
