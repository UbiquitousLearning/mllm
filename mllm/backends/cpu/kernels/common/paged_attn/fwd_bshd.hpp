// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/backends/cpu/kernels/common/paged_attn/arch.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

namespace mllm::cpu::paged_attn {

// BSHD
// K: [B, S(full block), H_KV, D]
// V: [B, S(full block), H_KV, D]
// Q: [B, S(partial block), H_Q, D]
// index: [S(partial block)]
//
// H_KV should <= H_Q
__MLLM_UNSAFE_OPT_BEGIN_O3
template<typename __ArchTag>
MLLM_FORCE_INLINE bool fwd_bshd_fp32_fastexp(int32_t B, int32_t S_Q, int32_t S_KV, int32_t H_Q, int32_t H_KV, int32_t D,
                                             const mllm_fp32_t* __restrict__ __q, const mllm_fp32_t* __restrict__ __k,
                                             const mllm_fp32_t* __restrict__ __v, mllm_fp32_t* __restrict__ __out,
                                             mllm_fp32_t* __restrict__ __attn_weights, const mllm_int32_t* __restrict__ __index,
                                             const mllm_fp32_t* __restrict__ __custom_causal_mask,
                                             bool need_return_attn_weights, int32_t thread_count) {
  if (D % 4 || H_KV > H_Q || H_Q % H_KV || !__index) { return false; }

  auto head_repeat_times = H_Q / H_KV;

  auto scale = 1.f / sqrtf(D);

  // 3 Passes.
  if (need_return_attn_weights) {
    if (!__attn_weights) { return false; }

    // Pass 1. Q @ K^T
    for (int32_t b_i = 0; b_i < B; ++b_i) {                                                  // B
      for (int32_t s_q_i = 0; s_q_i < S_Q; ++s_q_i) {                                        // S_Q
        for (int32_t s_kv_i = 0; s_kv_i < S_KV; ++s_kv_i) {                                  // S_KV
          MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_i, 0, H_Q, 1, {  // H(Parallel Loop at head level)
            int32_t h_kv_i = h_q_i / head_repeat_times;
            // Short cut. S_Q == 1.
            if (S_Q == 1 || __custom_causal_mask[s_q_i * S_KV + s_kv_i] > -1.f) {
              details::VectorDotProduct<__ArchTag, kFloat32, kFloat32, kFloat32>::run(
                  __q + b_i * H_Q * S_Q * D + s_q_i * H_Q * D + h_q_i * D,
                  __k + b_i * H_KV * S_KV * D + __index[s_kv_i] * H_KV * D + h_kv_i * D,
                  __attn_weights + b_i * H_Q * S_Q * S_KV + h_q_i * S_Q * S_KV + s_q_i * S_KV + s_kv_i, D);
              __attn_weights[b_i * H_Q * S_Q * S_KV + h_q_i * S_Q * S_KV + s_q_i * S_KV + s_kv_i] *= scale;
            } else {
              __attn_weights[b_i * H_Q * S_Q * S_KV + h_q_i * S_Q * S_KV + s_q_i * S_KV + s_kv_i] =
                  DataTypeInfo<mllm_fp32_t>::min();
            }
          });
        }
      }
    }

    // Pass 2. Softmax
    // Attn is [B, H, S_Q, S_KV] format
    for (int32_t b_i = 0; b_i < B; ++b_i) {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_i, 0, H_Q, 1, {
        for (int32_t s_q_i = 0; s_q_i < S_Q; ++s_q_i) {
          const int32_t offset = b_i * H_Q * S_Q * S_KV + h_q_i * S_Q * S_KV + s_q_i * S_KV;
          details::Softmax<__ArchTag, kFloat32, false>::run(__attn_weights + offset, S_KV);
        }
      });
    }

    // Pass 3. Weight @ V
    // If we need attention. The weight @ V is extremely slow here. We use extra_workspace to make V Tensor contiguous. But you
    // should keep in mind that getting attn from paged_attn is not a good idea.
    //
    // V is [B, S_Q, H_KV, D] format
    // Attn is [B, H_Q, S_Q, S_KV] foramt
    for (int32_t b_i = 0; b_i < B; ++b_i) {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_i, 0, H_Q, 1, {
        int32_t h_kv_i = h_q_i / head_repeat_times;
        for (int32_t s_q_i = 0; s_q_i < S_Q; ++s_q_i) {
          for (int32_t d_i = 0; d_i < D; d_i += 4) {
            // Initialize accumulator registers for vectorized output
#if defined(__SSE__) || defined(__x86_64__)
            __m128 acc = _mm_setzero_ps();
#elif defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t acc = vdupq_n_f32(0.0f);
#else
                // Fallback to scalar
                mllm_fp32_t acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif

            for (int32_t s_kv_i = 0; s_kv_i < S_KV; ++s_kv_i) {
              // Load attention weight
              mllm_fp32_t weight = __attn_weights[b_i * H_Q * S_Q * S_KV + h_q_i * S_Q * S_KV + s_q_i * S_KV + s_kv_i];

              // Load value vector (4 elements at a time)
              int32_t physical_block = __index[s_kv_i];
              const mllm_fp32_t* v_ptr = __v + b_i * H_KV * S_KV * D + physical_block * H_KV * D + h_kv_i * D + d_i;

#if defined(__SSE__) || defined(__x86_64__)
              __m128 weight_vec = _mm_set1_ps(weight);
              __m128 v_vec = _mm_load_ps(v_ptr);
              acc = _mm_add_ps(acc, _mm_mul_ps(weight_vec, v_vec));
#elif defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t weight_vec = vdupq_n_f32(weight);
                    float32x4_t v_vec = vld1q_f32(v_ptr);
                    acc = vaddq_f32(acc, vmulq_f32(weight_vec, v_vec));
#else
                    // Fallback to scalar
                    for (int i = 0; i < 4; i++) {
                        acc[i] += weight * v_ptr[i];
                    }
#endif
            }

            // Store result to [B, S, H, D]
            mllm_fp32_t* out_ptr = __out + b_i * H_Q * S_Q * D + s_q_i * H_Q * D + h_q_i * D + d_i;

#if defined(__SSE__) || defined(__x86_64__)
            _mm_store_ps(out_ptr, acc);
#elif defined(__ARM_NEON) || defined(__aarch64__)
                vst1q_f32(out_ptr, acc);
#else
                // Fallback to scalar
                for (int i = 0; i < 4; i++) {
                    out_ptr[i] = acc[i];
                }
#endif
          }
        }
      });
    }
  } else
  // 2 Pass FA2 Method.
  {
    for (int32_t b_i = 0; b_i < B; ++b_i) {
      // Parallel Loop at head level
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, h_q_i, 0, H_Q, 1, {
        int32_t h_kv_i = h_q_i / head_repeat_times;
        // TODO for loops here.
      });
    }
  }
  return true;
}
__MLLM_UNSAFE_OPT_END
}  // namespace mllm::cpu::paged_attn
