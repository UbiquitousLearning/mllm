// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/ops/PagedAttnOp.hpp"
#include "mllm/backends/cpu/kernels/common/paged_attn/fwd_bshd.hpp"

namespace mllm::cpu {

CPUPagedAttnOp::CPUPagedAttnOp(const aops::PagedAttnOpOptions& options) : aops::PagedAttnOp(options) {}

void CPUPagedAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& index = inputs[3];
  auto& causal_mask = inputs[4];

  auto q_shape = Q.shape();
  auto B = q_shape[0];
  auto S_Q = q_shape[1];
  auto S_KV = index.shape()[0];
  auto H_Q = q_shape[2];
  auto H_KV = H_Q / options_.head_repeat_times;
  auto D = q_shape[3];

  auto& out = outputs[0];
  Tensor attn = options_.need_attn_weights ? outputs[1] : Tensor::nil();

  // Not support yet.
  MLLM_RT_ASSERT_EQ(options_.fuse_rope, false);

  switch (options_.impl_type) {
    case aops::PagedAttnImplType::kAllFp32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      paged_attn::fwd_bshd_fp32_fastexp<paged_attn::details::arm_arch_tag>(
          B, S_Q, S_KV, H_Q, H_KV, D, Q.ptr<mllm_fp32_t>(), K.ptr<mllm_fp32_t>(), V.ptr<mllm_fp32_t>(), out.ptr<mllm_fp32_t>(),
          options_.need_attn_weights ? attn.ptr<mllm_fp32_t>() : nullptr, index.ptr<int32_t>(),
          S_Q == 1 ? nullptr : causal_mask.ptr<mllm_fp32_t>(), options_.need_attn_weights, options_.getThreads());
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      paged_attn::fwd_bshd_fp32_fastexp<paged_attn::details::arm_arch_tag>(
          B, S_Q, S_KV, H_Q, H_KV, D, Q.ptr<mllm_fp32_t>(), K.ptr<mllm_fp32_t>(), V.ptr<mllm_fp32_t>(), out.ptr<mllm_fp32_t>(),
          options_.need_attn_weights ? attn.ptr<mllm_fp32_t>() : nullptr, index.ptr<int32_t>(),
          S_Q == 1 ? nullptr : causal_mask.ptr<mllm_fp32_t>(), options_.need_attn_weights, options_.getThreads());
#else
      paged_attn::fwd_bshd_fp32_fastexp<paged_attn::details::arm_arch_tag>(
          B, S_Q, S_KV, H_Q, H_KV, D, Q.ptr<mllm_fp32_t>(), K.ptr<mllm_fp32_t>(), V.ptr<mllm_fp32_t>(), out.ptr<mllm_fp32_t>(),
          options_.need_attn_weights ? attn.ptr<mllm_fp32_t>() : nullptr, index.ptr<int32_t>(),
          S_Q == 1 ? nullptr : causal_mask.ptr<mllm_fp32_t>(), options_.need_attn_weights, options_.getThreads());
#endif
      break;
    }
    case aops::PagedAttnImplType::kDefault: {
      break;
    }
  }

  // TODO
}

}  // namespace mllm::cpu
