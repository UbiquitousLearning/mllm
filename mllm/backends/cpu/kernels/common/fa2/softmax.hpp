// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/utils/Common.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/primitives.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#include "mllm/backends/cpu/kernels/arm/math.hpp"
#if defined(MLLM_HOST_FEATURE_FP16)
#include <arm_fp16.h>
#endif  // defined(MLLM_HOST_FEATURE_FP16)
#endif  // defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::fa2 {

//===----------------------------------------------------------------------===//
// General Template
//===----------------------------------------------------------------------===//
template<typename ArchTrait_, typename TileTrait_, typename NumericTrait_, typename MemoryTrait_, bool HP_>
struct Softmax {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                    ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                    ElementAccumulator* score_scale, ElementAccumulator* score_sum, ElementAccumulator* logsum,
                                    ElementAccumulator scale) {}

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                         ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                         ElementAccumulator* score_scale, ElementAccumulator* score_sum,
                                         ElementAccumulator* logsum, ElementAccumulator scale) {}
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#if defined(MLLM_HOST_FEATURE_FP16)

// clang-format off
template<int M_, int N_, bool HP_>
struct Softmax<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<mllm_fp32_t, mllm_fp16_t>,
    MemoryTrait<128>,
    HP_
>
// clang-format on
{
  static constexpr ArchType kArch = ArmNeon128ArchTrait::kArch;

  static constexpr int kTileM = TileTrait<M_, N_, -1>::kTileM;
  static constexpr int kTileN = TileTrait<M_, N_, -1>::kTileN;
  static constexpr int kTileK = TileTrait<M_, N_, -1>::kTileK;

  using ElementAccumulator = NumericTrait<mllm_fp32_t, mllm_fp16_t>::ElementAccumulator;
  using ElementCompute = NumericTrait<mllm_fp32_t, mllm_fp16_t>::ElementCompute;

  static constexpr int kAlignment = MemoryTrait<128>::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait<128>::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                    ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                    ElementAccumulator* score_scale, ElementAccumulator* score_sum, ElementAccumulator* logsum,
                                    ElementAccumulator scale) {
    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, kTileM * sizeof(ElementAccumulator));

// 2. Reduce max(acc_s) to scoremax
#pragma unroll
    for (int br = 0; br < kTileM; ++br) {
      float32x4_t max_vec = vdupq_n_f32(FA2_FLOAT_NEG_INF);
      const ElementAccumulator* row = acc_s + br * kTileN;
// Vectorized max reduction
#pragma unroll
      for (int bc = 0; bc < kTileN; bc += 4) {
        float32x4_t vals = vld1q_f32(row + bc);
        max_vec = vmaxq_f32(max_vec, vals);
      }
      // Handle remaining elements
      float max_val = vmaxvq_f32(max_vec);
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    float32x4_t scale_vec = vdupq_n_f32(scale);
    if constexpr (!HP_) {  // Use approximate method to calculate exp.
#pragma unroll
      for (int br = 0; br < kTileM; br += 4) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, arm::vexpq_fast_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    } else {  // High precession exp use libcall function from cmath.
#pragma unroll
      for (int br = 0; br < kTileM; ++br) {
        float32x4_t smp_vec = vld1q_f32(scoremax_prev + br);
        float32x4_t sm_vec = vld1q_f32(scoremax + br);
        vst1q_f32(score_scale + br, arm::vexpq_hp_f32(vmulq_f32(vsubq_f32(smp_vec, sm_vec), scale_vec)));
      }
    }

// 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
#pragma unroll
    for (int br = 0; br < kTileM; ++br) {
      const float sm = scoremax[br];
      ElementAccumulator* row = const_cast<ElementAccumulator*>(acc_s) + br * kTileN;

      float sum = 0.0f;
      const float32x4_t sm_vec = vdupq_n_f32(sm);

      // Vectorized processing
      for (int bc = 0; bc < kTileN; bc += 4) {
        float32x4_t val_vec = vld1q_f32(row + bc);

        // Compute: (val - sm) * scale
        val_vec = vsubq_f32(val_vec, sm_vec);
        val_vec = vmulq_f32(val_vec, scale_vec);

        // Vectorized exp calculation
        float32x4_t exp_vec;
        if constexpr (!HP_) {
          exp_vec = arm::vexpq_fast_f32(val_vec);
        } else {
          exp_vec = arm::vexpq_hp_f32(val_vec);
        }

        // Store results
        vst1q_f32(row + bc, exp_vec);

        // Accumulate sum
        sum += vaddvq_f32(exp_vec);
      }
      score_sum[br] = sum;
    }

// 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
#pragma unroll
    for (int br = 0; br < kTileM; br += 4) {
      float32x4_t logsum_v = vld1q_f32(logsum + br);
      float32x4_t scale_v = vld1q_f32(score_scale + br);
      float32x4_t sum_v = vld1q_f32(score_sum + br);
      vst1q_f32(logsum + br, vmlaq_f32(sum_v, logsum_v, scale_v));
    }

// 7. Copy acc_s to acc_s_cast
#pragma unroll
    for (int i = 0; i < kTileM * kTileN; i += 8) {
      float32x4_t v0 = vld1q_f32(acc_s + i);
      float32x4_t v1 = vld1q_f32(acc_s + i + 4);
      float16x8_t v16 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
      vst1q_f16(acc_s_cast + i, v16);
    }
  }

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                         ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                         ElementAccumulator* score_scale, ElementAccumulator* score_sum,
                                         ElementAccumulator* logsum, ElementAccumulator scale) {
    // 1. Copy scoremax to scoremax_prev
    memcpy(scoremax_prev, scoremax, real_m_block_size * sizeof(ElementAccumulator));

    // 2. Reduce max(acc_s) to scoremax
    for (int br = 0; br < real_m_block_size; ++br) {
      const ElementAccumulator* row = acc_s + br * kTileN;
      float max_val = -std::numeric_limits<float>::infinity();
      for (int bc = 0; bc < real_n_block_size; ++bc) { max_val = std::max(max_val, row[bc]); }
      scoremax[br] = max_val;
    }

    // 3. scores_scale[i] = exp(scores_max_prev[i] * scale - scores_max[i] * scale)
    for (int br = 0; br < real_m_block_size; ++br) { score_scale[br] = std::exp((scoremax_prev[br] - scoremax[br]) * scale); }

    // 4. acc_s[i, j] = exp(acc_s[i, j] * scale - scores_max[i] * scale) and update score_sum
    for (int br = 0; br < real_m_block_size; ++br) {
      const float sm = scoremax[br];
      ElementAccumulator* row = const_cast<ElementAccumulator*>(acc_s) + br * kTileN;
      float sum = 0.0f;

      for (int bc = 0; bc < real_n_block_size; ++bc) {
        const float val = std::exp((row[bc] - sm) * scale);
        row[bc] = val;
        sum += val;
      }
      score_sum[br] = sum;
    }

    // 6. logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    for (int br = 0; br < real_m_block_size; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }

    // 7. Copy acc_s to acc_s_cast
    for (int br = 0; br < real_m_block_size; ++br) {
      const ElementAccumulator* acc_s_row = acc_s + br * kTileN;
      ElementCompute* acc_s_cast_row = acc_s_cast + br * kTileN;
      for (int bc = 0; bc < real_n_block_size; ++bc) { acc_s_cast_row[bc] = static_cast<ElementCompute>(acc_s_row[bc]); }
    }
  }
};

#endif
#endif
}  // namespace mllm::cpu::fa2
