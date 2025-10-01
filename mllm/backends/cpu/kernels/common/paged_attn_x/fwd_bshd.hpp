// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/Parallel.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/paged_attn_x/arch.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
#include "mllm/backends/cpu/kernels/common/paged_attn_x/impl-arm.hpp"
#else
#include "mllm/backends/cpu/kernels/common/paged_attn_x/impl-any-simd.hpp"
#endif
#include "mllm/backends/cpu/kernels/common/paged_attn_x/impl-any.hpp"

namespace mllm::cpu::paged_attn_x {

// BSHD
// K: [B, H_KV, S_KV, D], not contiguous
// V: [B, H_KV, S_KV, D], not contiguous
// Q: [B, H_Q, S_Q, D], contiguous
//
// H_KV should <= H_Q
template<typename __ArchTag, typename __QDType, typename __KDType, typename __VDType, typename __ODType, typename __AccDType,
         bool high_precession_exp = true>
void fwd_bhsd(int32_t B, int32_t H_Q, int32_t H_KV, int32_t S_Q, int32_t S_KV, int32_t D, const __QDType* __restrict__ __q,
              const __KDType** __restrict__ __k, const __VDType** __restrict__ __v, __ODType* __restrict__ __out,
              int32_t thread_count) {
  int32_t head_repeat_times = H_Q / H_KV;

  // Loop on batch size.
  for (int b_idx = 0; b_idx < B; ++b_idx) {
    // Loop on head dim, should be made parallel
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_idx, 0, H_Q, 1, {
      int h_kv_id = h_q_idx / head_repeat_times;

      // FA2's Loop
      for (int s_q_idx = 0; s_q_idx < S_Q; ++s_q_idx) {
        __QDType* q_token = __q + b_idx * H_Q * S_Q * D + h_q_idx * S_Q * D + s_q_idx * D;
        __ODType* o_token = __out + b_idx * H_Q * S_Q * D + h_q_idx * S_Q * D + s_q_idx * D;

        __AccDType score_v;
        __AccDType max_v;
        __AccDType out_v;
        __AccDType scale_v;

        for (int s_kv_idx = 0; s_kv_idx < S_KV; ++s_kv_idx) {
          // TODO, prefetch next

          __KDType* k_token = *(__k + s_kv_idx);
          __VDType* v_token = *(__v + s_kv_idx);

          // Offset to one head.
          // k_token and v_token shape is [D]
          k_token = k_token + b_idx * H_KV * D + h_kv_id * D;
          v_token = v_token + b_idx * H_KV * D + h_kv_id * D;

          // TODO
          // If masked. return.

          // 1. MMA0. Q @ K -> A_i
          __ODType a;
          details::VectorDotProduct<__ArchTag, __QDType, __KDType, __AccDType>::run(q_token, k_token, &a, D);

          // 2. Do softmax stuff.
          // TODO

          // 3. Scale
          // TODO

          // 4. MMA1.
          // TODO

          // TODO, drop in the feature.
        }

        // 5. Final Rescale.
        // TODO

        // If masked, Set -inf.
        // TODO
      }
    });
  }
}

}  // namespace mllm::cpu::paged_attn_x
