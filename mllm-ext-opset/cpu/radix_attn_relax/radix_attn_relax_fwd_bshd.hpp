// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <numbers>
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/arch.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH)
#include "mllm/backends/cpu/kernels/common/radix_attn/impl-arm.hpp"
#else
#include "mllm/backends/cpu/kernels/common/radix_attn/impl-any-simd.hpp"
#endif
#include "mllm/backends/cpu/kernels/common/radix_attn/impl-any.hpp"

// BSHD
// K: [S_KV], address, not contiguous
// V: [S_KV], address, not contiguous
// Q: [B, S_Q, H_Q, D], contiguous
//
// After find KV Tokens, KV is [B, 1, H_KV, D]
//
// H_KV should <= H_Q
template<typename __ArchTag, typename __QDType, typename __KDType, typename __VDType, typename __ODType, typename __AccDType,
         bool high_precession_exp = true>
void fwd_bshd(int32_t B, int32_t H_Q, int32_t H_KV, int32_t S_Q, int32_t S_KV, int32_t D_QK, int32_t D_V,
              const __QDType* __restrict__ __q, __KDType** __k, __VDType** __v, __ODType* __restrict__ __out,
              int32_t thread_count) {
  int32_t head_repeat_times = H_Q / H_KV;

  __AccDType scale = scale = std::sqrt(1.0 / D_QK) * (__AccDType)std::numbers::log2e;

  // Loop on batch size.
  for (int b_idx = 0; b_idx < B; ++b_idx) {
    // Loop on HEAD dim, should be made parallel
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_idx, 0, H_Q, 1, {
      int h_kv_id = h_q_idx / head_repeat_times;

      // FA2's Loop
      for (int s_q_idx = 0; s_q_idx < S_Q; ++s_q_idx) {
        const __QDType* q_token = __q + b_idx * H_Q * S_Q * D_QK + s_q_idx * H_Q * D_QK + h_q_idx * D_QK;
        __ODType* acc_o = __out + b_idx * H_Q * S_Q * D_V + s_q_idx * H_Q * D_V + h_q_idx * D_V;

        mllm::cpu::radix_attn::details::FilledWithConst<__ArchTag, __ODType>::run(acc_o, 0, D_V);

        __AccDType scores_max = -std::numeric_limits<__AccDType>::infinity();
        __AccDType scores_max_prev = -std::numeric_limits<__AccDType>::infinity();
        __AccDType logsum = 0;
        __AccDType scores_sum = 0;
        __AccDType scores_scale = 0;

        int __delta = S_KV - S_Q;
        int S_KV_BOUND = std::min(__delta + s_q_idx + 1, S_KV);

        for (int s_kv_idx = 0; s_kv_idx < S_KV_BOUND; ++s_kv_idx) {
          // k_token and v_token shape is [B, 1, H, D]
          __KDType* k_token = __k[s_kv_idx];
          __VDType* v_token = __v[s_kv_idx];

          // Offset to one head.
          // k_token and v_token shape is [D]
          k_token = k_token + b_idx * H_KV * D_QK + h_kv_id * D_QK;
          v_token = v_token + b_idx * H_KV * D_V + h_kv_id * D_V;

          // 1. MMA0. Q @ K -> A_i
          __AccDType acc_s;
          mllm::cpu::radix_attn::details::VectorDotProduct<__ArchTag, __QDType, __KDType, __AccDType>::run(q_token, k_token,
                                                                                                           &acc_s, D_QK);

          // 2. Do softmax stuff.
          scores_max_prev = scores_max;
          scores_max = std::max(scores_max_prev, acc_s);
          scores_scale = std::exp2(scores_max_prev * scale - scores_max * scale);
          acc_s = std::exp2(acc_s * scale - scores_max * scale);
          scores_sum = acc_s;
          logsum = logsum * scores_scale + scores_sum;

          // 3. Scale
          mllm::cpu::radix_attn::details::MulFromConst<__ArchTag, __AccDType, __AccDType>::run(acc_o, scores_scale, D_V);

          // 4. MMA1.
          mllm::cpu::radix_attn::details::FMAConstArray<__ArchTag, __AccDType, __AccDType, __AccDType>::run(acc_o, acc_s,
                                                                                                            v_token, D_V);
        }

        // 5. Final Rescale.
        mllm::cpu::radix_attn::details::MulFromConst<__ArchTag, __ODType, __AccDType>::run(acc_o, (1.f / logsum), D_V);
      }
    });
  }
}
