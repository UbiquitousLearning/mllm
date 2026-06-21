// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

// We use the same logic of flash_attention_2_1 here.
#include <cmath>
#include <limits>
#include <cstdint>
#include <numbers>
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/fa2_1/arch.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-arm.hpp"
#else
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any-simd.hpp"
#endif
#include "mllm/backends/cpu/kernels/common/fa2_1/impl-any.hpp"

// QKV is BSHD
// H_KV should <= H_Q
template<typename __ArchTag, typename __QDType, typename __KDType, typename __VDType, typename __ODType, typename __AccDType,
         bool high_precession_exp = true>
void fwd_bshd_swa_with_sink(int32_t B, int32_t H_Q, int32_t H_KV, int32_t S_Q, int32_t S_KV, int32_t D_QK, int32_t D_V,
                            int left_sliding_window, int32_t cur_kv_seq_len, const __QDType* __restrict__ __q, __KDType* __k,
                            __VDType* __v, __AccDType* __s_aux, __ODType* __restrict__ __out, int32_t thread_count) {
  // NOTE: Assume s_kv always less equal then sliding_window.
  if (S_KV > left_sliding_window) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCPUKernelError,
                    "Assume s_kv always less equal then sliding_window in fwd_bshd_swa_with_sink kernel.");
  }

  int32_t head_repeat_times = H_Q / H_KV;

  __AccDType scale = std::sqrt(1.0 / D_QK) * (__AccDType)std::numbers::log2e;

  // Get cur window size
  auto cur_window_size = std::min(left_sliding_window, S_KV);

  // Loop on batch size.
  for (int b_idx = 0; b_idx < B; ++b_idx) {
    // Loop on HEAD dim, should be made parallel
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_idx, 0, H_Q, 1, {
      int h_kv_id = h_q_idx / head_repeat_times;

      // FA2's Loop
      for (int s_q_idx = 0; s_q_idx < S_Q; ++s_q_idx) {
        const __QDType* q_token = __q + b_idx * H_Q * S_Q * D_QK + s_q_idx * H_Q * D_QK + h_q_idx * D_QK;
        __ODType* acc_o = __out + b_idx * H_Q * S_Q * D_V + s_q_idx * H_Q * D_V + h_q_idx * D_V;

        mllm::cpu::flash_attn2::details::FilledWithConst<__ArchTag, __ODType>::run(acc_o, 0, D_V);

        __AccDType scores_max = -std::numeric_limits<__AccDType>::infinity();
        __AccDType scores_max_prev = -std::numeric_limits<__AccDType>::infinity();
        __AccDType logsum = 0;
        __AccDType scores_sum = 0;
        __AccDType scores_scale = 0;

        // Compute should be done in [full_attention_start, full_attention_end]
        auto full_attention_start = std::max(0, s_q_idx + cur_kv_seq_len - S_Q - left_sliding_window);
        auto full_attention_end = std::min(cur_kv_seq_len, s_q_idx + cur_kv_seq_len - S_Q);

        // We then map full attention to local attention position;
        auto local_attention_start = 0;

        auto local_attention_end = full_attention_end - full_attention_start;
        if (local_attention_end < left_sliding_window) local_attention_end++;

        for (int s_kv_idx = local_attention_start; s_kv_idx < local_attention_end; ++s_kv_idx) {
          // k_token and v_token shape is [B, 1, H, D]
          __KDType* k_token = __k + b_idx * S_KV * H_KV * D_QK + s_kv_idx * H_KV * D_QK;
          __VDType* v_token = __v + b_idx * S_KV * H_KV * D_V + s_kv_idx * H_KV * D_V;

          // Offset to one head.
          // k_token and v_token shape is [D]
          k_token = k_token + h_kv_id * D_QK;
          v_token = v_token + h_kv_id * D_V;

          // 1. MMA0. Q @ K -> A_i
          __AccDType acc_s;
          mllm::cpu::flash_attn2::details::VectorDotProduct<__ArchTag, __QDType, __KDType, __AccDType>::run(q_token, k_token,
                                                                                                            &acc_s, D_QK);

          // 2. Do softmax stuff.
          scores_max_prev = scores_max;
          scores_max = std::max(scores_max_prev, acc_s);
          scores_scale = std::exp2(scores_max_prev * scale - scores_max * scale);
          acc_s = std::exp2(acc_s * scale - scores_max * scale);
          scores_sum = acc_s;
          logsum = logsum * scores_scale + scores_sum;

          // 3. Scale
          mllm::cpu::flash_attn2::details::MulFromConst<__ArchTag, __AccDType, __AccDType>::run(acc_o, scores_scale, D_V);

          // 4. MMA1.
          mllm::cpu::flash_attn2::details::FMAConstArray<__ArchTag, __AccDType, __AccDType, __AccDType>::run(acc_o, acc_s,
                                                                                                             v_token, D_V);
        }

        // We attach virtual sink token first for all query loops
        __AccDType sink_v = __s_aux[h_q_idx];

        // sink_token does not need 1/sqrt(D) scale, we divide it by 1/sqrt(D) scale first. Then we can reuse the logic in
        // online-softmax
        sink_v /= std::sqrt(1.0 / D_QK);

        // Do softmax stuff on this sink token value.
        scores_max_prev = scores_max;
        scores_max = std::max(scores_max_prev, sink_v);
        scores_scale = std::exp2(scores_max_prev * scale - scores_max * scale);
        sink_v = std::exp2(sink_v * scale - scores_max * scale);
        scores_sum = sink_v;
        logsum = logsum * scores_scale + scores_sum;
        mllm::cpu::flash_attn2::details::MulFromConst<__ArchTag, __AccDType, __AccDType>::run(acc_o, scores_scale, D_V);

        // 5. Final Rescale.
        mllm::cpu::flash_attn2::details::MulFromConst<__ArchTag, __ODType, __AccDType>::run(acc_o, (1.f / logsum), D_V);
      }
    });
  }
}
