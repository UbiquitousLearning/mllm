// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstddef>
#include <cstring>

#include "mllm/backends/cpu/ops/CausalMaskOp.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#include <immintrin.h>  // Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

CPUCausalMaskOp::CPUCausalMaskOp(const aops::CausalMaskOpOptions& options) : aops::CausalMaskOp(options) {}

void CPUCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  auto shape = ins.shape();

  auto B = shape[0];
  auto H = shape[1];
  auto S = shape[2];
  // For self-attention, the last two dimensions are sequence length (S) x sequence length (S).
  // We assume D == S.
  auto D = shape[3];

  switch (ins.dtype()) {
    case kFloat32: {
      if (S == 1) {  // When sequence length is 1, no masking is needed.
        for (int b = 0; b < B; ++b) {
          for (int h = 0; h < H; ++h) {
            auto* i_ptr = ins.offsettedPtr<float>({b, h, 0, 0});
            auto* o_ptr = ous.offsettedPtr<float>({b, h, 0, 0});
            memcpy(o_ptr, i_ptr, D * sizeof(float));
          }
        }
        return;
      }

      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          auto* i_ptr = ins.offsettedPtr<float>({b, h, 0, 0});
          auto* o_ptr = ous.offsettedPtr<float>({b, h, 0, 0});

          if (!options_.sliding_window) {
            // Standard causal mask
#if (defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)) && defined(__AVX2__)
            const __m256 mask_val = _mm256_set1_ps(-1e10f);
            for (size_t r = 0; r < S; ++r) {
              const size_t row_offset = r * D;
              const size_t copy_count = D - S + r + 1;
              const size_t fill_count = std::max(D - copy_count, (size_t)0);

              memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(float));

              float* fill_start = o_ptr + row_offset + copy_count;
              size_t avx_iters = fill_count / 8;
              size_t remainder = fill_count % 8;

              for (size_t i = 0; i < avx_iters; ++i) { _mm256_storeu_ps(fill_start + i * 8, mask_val); }
              for (size_t i = 0; i < remainder; ++i) { fill_start[avx_iters * 8 + i] = -1e10f; }
            }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
            const float32x4_t mask_val = vdupq_n_f32(-1e10f);
            for (size_t r = 0; r < S; ++r) {
              const size_t copy_count = D - S + r + 1;
              const size_t fill_count = std::max(D - copy_count, (size_t)0);

              memcpy(o_ptr + r * D, i_ptr + r * D, copy_count * sizeof(float));

              float* fill_start = o_ptr + r * D + copy_count;

              size_t neon_iters = fill_count / 4;
              size_t remainder = fill_count % 4;

              for (size_t i = 0; i < neon_iters; ++i) { vst1q_f32(fill_start + i * 4, mask_val); }
              for (size_t i = 0; i < remainder; ++i) { fill_start[neon_iters * 4 + i] = -1e10f; }
            }
#else
            for (size_t r = 0; r < S; ++r) {
              const size_t row_offset = r * D;
              const size_t copy_count = D - S + r + 1;
              const size_t fill_count = std::max(D - copy_count, (size_t)0);

              memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(float));

              float* fill_start = o_ptr + row_offset + copy_count;
              for (size_t i = 0; i < fill_count; ++i) { fill_start[i] = -1e10f; }
            }
#endif
          } else {
            // Sliding window causal mask
            const int window_size = options_.window_size;
            for (int s = 0; s < S; ++s) {
              const size_t row_offset = s * S;
              const int copy_start_idx = std::max(0, s - window_size + 1);

              // 1. Mask prefix
              const size_t prefix_fill_count = copy_start_idx;
              // 2. Copy content
              const size_t copy_count = s - copy_start_idx + 1;
              memcpy(o_ptr + row_offset + copy_start_idx, i_ptr + row_offset + copy_start_idx, copy_count * sizeof(float));
              // 3. Mask suffix
              const size_t suffix_fill_start_idx = s + 1;
              const size_t suffix_fill_count = S - suffix_fill_start_idx;

#if (defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)) && defined(__AVX2__)
              const __m256 mask_val = _mm256_set1_ps(-1e10f);
              // Fill prefix
              float* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count / 8; ++i) _mm256_storeu_ps(prefix_fill_start + i * 8, mask_val);
              for (size_t i = (prefix_fill_count / 8) * 8; i < prefix_fill_count; ++i) prefix_fill_start[i] = -1e10f;
              // Fill suffix
              float* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count / 8; ++i) _mm256_storeu_ps(suffix_fill_start + i * 8, mask_val);
              for (size_t i = (suffix_fill_count / 8) * 8; i < suffix_fill_count; ++i) suffix_fill_start[i] = -1e10f;
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
              const float32x4_t mask_val = vdupq_n_f32(-1e10f);
              // Fill prefix
              float* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count / 4; ++i) vst1q_f32(prefix_fill_start + i * 4, mask_val);
              for (size_t i = (prefix_fill_count / 4) * 4; i < prefix_fill_count; ++i) prefix_fill_start[i] = -1e10f;
              // Fill suffix
              float* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count / 4; ++i) vst1q_f32(suffix_fill_start + i * 4, mask_val);
              for (size_t i = (suffix_fill_count / 4) * 4; i < suffix_fill_count; ++i) suffix_fill_start[i] = -1e10f;
#else
              float* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count; ++i) { prefix_fill_start[i] = -1e10f; }
              float* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count; ++i) { suffix_fill_start[i] = -1e10f; }
#endif
            }
          }
        }
      }
      break;
    }
    case kFloat16: {
      if (S == 1) {  // When sequence length is 1, no masking is needed.
        for (int b = 0; b < B; ++b) {
          for (int h = 0; h < H; ++h) {
            auto* i_ptr = ins.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});
            auto* o_ptr = ous.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});
            memcpy(o_ptr, i_ptr, D * sizeof(mllm_fp16_t));
          }
        }
        return;
      }
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          auto* i_ptr = ins.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});
          auto* o_ptr = ous.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});

          if (!options_.sliding_window) {
            // Standard causal mask
#if (defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)) && defined(__AVX2__) && defined(__F16C__)
            const __m256 mask_ps = _mm256_set1_ps(-65500.f);
            const __m128i mask_val = _mm256_cvtps_ph(mask_ps, _MM_FROUND_TO_NEAREST_INT);
            for (size_t s = 0; s < S; ++s) {
              const size_t row_offset = s * S;
              const size_t copy_count = s + 1;
              const size_t fill_count = S - copy_count;

              if (copy_count > 0) { memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(mllm_fp16_t)); }

              mllm_fp16_t* fill_start = o_ptr + row_offset + copy_count;
              size_t avx_iters = fill_count / 8;
              size_t remainder = fill_count % 8;

              for (size_t i = 0; i < avx_iters; ++i) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(fill_start + i * 8), mask_val);
              }
              for (size_t i = 0; i < remainder; ++i) { fill_start[avx_iters * 8 + i] = -65500.f; }
            }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
            const float16x8_t mask_val = vdupq_n_f16(-65500.f);
            for (size_t s = 0; s < S; ++s) {
              const size_t row_offset = s * S;
              const size_t copy_count = s + 1;
              const size_t fill_count = S - copy_count;

              if (copy_count > 0) { memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(mllm_fp16_t)); }

              mllm_fp16_t* fill_start = o_ptr + row_offset + copy_count;
              size_t neon_iters = fill_count / 8;
              size_t remainder = fill_count % 8;

              for (size_t i = 0; i < neon_iters; ++i) { vst1q_f16(fill_start + i * 8, mask_val); }
              for (size_t i = 0; i < remainder; ++i) { fill_start[neon_iters * 8 + i] = -65500.f; }
            }
#else
            for (size_t s = 0; s < S; ++s) {
              const size_t row_offset = s * S;
              const size_t copy_count = s + 1;
              const size_t fill_count = S - copy_count;

              if (copy_count > 0) { memcpy(o_ptr + row_offset, i_ptr + row_offset, copy_count * sizeof(mllm_fp16_t)); }

              mllm_fp16_t* fill_start = o_ptr + row_offset + copy_count;
              for (size_t i = 0; i < fill_count; ++i) { fill_start[i] = -65500.f; }
            }
#endif
          } else {
            // Sliding window causal mask
            const int window_size = options_.window_size;
            for (int s = 0; s < S; ++s) {
              const size_t row_offset = s * S;
              const int copy_start_idx = std::max(0, s - window_size + 1);

              // 1. Mask prefix
              const size_t prefix_fill_count = copy_start_idx;
              // 2. Copy content
              const size_t copy_count = s - copy_start_idx + 1;
              memcpy(o_ptr + row_offset + copy_start_idx, i_ptr + row_offset + copy_start_idx,
                     copy_count * sizeof(mllm_fp16_t));
              // 3. Mask suffix
              const size_t suffix_fill_start_idx = s + 1;
              const size_t suffix_fill_count = S - suffix_fill_start_idx;

#if (defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)) && defined(__AVX2__) && defined(__F16C__)
              const __m256 mask_ps = _mm256_set1_ps(-65500.f);
              const __m128i mask_val = _mm256_cvtps_ph(mask_ps, _MM_FROUND_TO_NEAREST_INT);

              // Fill prefix
              mllm_fp16_t* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count / 8; ++i) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(prefix_fill_start + i * 8), mask_val);
              }
              for (size_t i = (prefix_fill_count / 8) * 8; i < prefix_fill_count; ++i) prefix_fill_start[i] = -65500.f;
              // Fill suffix
              mllm_fp16_t* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count / 8; ++i) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(suffix_fill_start + i * 8), mask_val);
              }
              for (size_t i = (suffix_fill_count / 8) * 8; i < suffix_fill_count; ++i) suffix_fill_start[i] = -65500.f;
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
              const float16x8_t mask_val = vdupq_n_f16(-65500.f);
              // Fill prefix
              mllm_fp16_t* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count / 8; ++i) vst1q_f16(prefix_fill_start + i * 8, mask_val);
              for (size_t i = (prefix_fill_count / 8) * 8; i < prefix_fill_count; ++i) prefix_fill_start[i] = -65500.f;
              // Fill suffix
              mllm_fp16_t* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count / 8; ++i) vst1q_f16(suffix_fill_start + i * 8, mask_val);
              for (size_t i = (suffix_fill_count / 8) * 8; i < suffix_fill_count; ++i) suffix_fill_start[i] = -65500.f;
#else
              mllm_fp16_t* prefix_fill_start = o_ptr + row_offset;
              for (size_t i = 0; i < prefix_fill_count; ++i) { prefix_fill_start[i] = -65500.f; }
              mllm_fp16_t* suffix_fill_start = o_ptr + row_offset + suffix_fill_start_idx;
              for (size_t i = 0; i < suffix_fill_count; ++i) { suffix_fill_start[i] = -65500.f; }
#endif
            }
          }
        }
      }
      break;
    }
    default: NYI("CausalMaskOp::forward just support fp32 and fp16 inputs right now");
  }
}
}  // namespace mllm::cpu
