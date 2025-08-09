// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/utils/Common.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/primitives.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#if defined(MLLM_HOST_FEATURE_FP16)
#include <arm_fp16.h>
#endif  // defined(MLLM_HOST_FEATURE_FP16)
#endif  // defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::fa2 {

//===----------------------------------------------------------------------===//
// General Template
//===----------------------------------------------------------------------===//
template<typename ArchTrait_, typename TileTrait_, typename NumericTrait_, typename MemoryTrait_>
struct ScaleCastCopy {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementAccumulator* __restrict__ acc_o, const ElementAccumulator* __restrict__ logsum,
                                    ElementCompute* __restrict__ o_block, const int32_t head_size, const int32_t dim_size) {}

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         const ElementAccumulator* __restrict__ acc_o,
                                         const ElementAccumulator* __restrict__ logsum, ElementCompute* __restrict__ o_block,
                                         const int32_t head_size, const int32_t dim_size) {}
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#if defined(MLLM_HOST_FEATURE_FP16)

// clang-format off
template<int M_, int N_>
struct ScaleCastCopy<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<mllm_fp32_t, mllm_fp16_t>,
    MemoryTrait<128>
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

  static MLLM_FORCE_INLINE void run(const ElementAccumulator* __restrict__ acc_o, const ElementAccumulator* __restrict__ logsum,
                                    ElementCompute* __restrict__ o_block, const int32_t head_size, const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < kTileM; ++i) {
      const float* acc_o_line = acc_o + i * dim_size;
      float16_t* o_block_line = o_block + i * head_size * dim_size;

      const float reciprocal_logsum = 1.0f / logsum[i];
      const float32x4_t vec_reciprocal = vdupq_n_f32(reciprocal_logsum);

      int j = 0;
      for (; j <= dim_size - 16; j += 16) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o_line + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o_line + j + 4);
        float32x4_t vec_acc_o_3 = vld1q_f32(acc_o_line + j + 8);
        float32x4_t vec_acc_o_4 = vld1q_f32(acc_o_line + j + 12);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal);
        float32x4_t result_vec_3 = vmulq_f32(vec_acc_o_3, vec_reciprocal);
        float32x4_t result_vec_4 = vmulq_f32(vec_acc_o_4, vec_reciprocal);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);
        float16x4_t result_half_3 = vcvt_f16_f32(result_vec_3);
        float16x4_t result_half_4 = vcvt_f16_f32(result_vec_4);

        float16x8_t result_half_12 = vcombine_f16(result_half_1, result_half_2);
        float16x8_t result_half_34 = vcombine_f16(result_half_3, result_half_4);

        vst1q_f16(o_block_line + j, result_half_12);
        vst1q_f16(o_block_line + j + 8, result_half_34);
      }

      if (j <= dim_size - 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o_line + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o_line + j + 4);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
        j += 8;
      }

      for (; j < dim_size; ++j) { o_block_line[j] = (float16_t)(acc_o_line[j] * reciprocal_logsum); }
    }
  }

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         const ElementAccumulator* __restrict__ acc_o,
                                         const ElementAccumulator* __restrict__ logsum, ElementCompute* __restrict__ o_block,
                                         const int32_t head_size, const int32_t dim_size) {
    for (int i = 0; i < real_m_block_size; ++i) {
      const float* acc_o_line = acc_o + i * dim_size;
      float16_t* o_block_line = o_block + i * head_size * dim_size;

      const float reciprocal_logsum = 1.0f / logsum[i];
      const float32x4_t vec_reciprocal = vdupq_n_f32(reciprocal_logsum);

      int j = 0;
      for (; j <= dim_size - 16; j += 16) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o_line + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o_line + j + 4);
        float32x4_t vec_acc_o_3 = vld1q_f32(acc_o_line + j + 8);
        float32x4_t vec_acc_o_4 = vld1q_f32(acc_o_line + j + 12);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal);
        float32x4_t result_vec_3 = vmulq_f32(vec_acc_o_3, vec_reciprocal);
        float32x4_t result_vec_4 = vmulq_f32(vec_acc_o_4, vec_reciprocal);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);
        float16x4_t result_half_3 = vcvt_f16_f32(result_vec_3);
        float16x4_t result_half_4 = vcvt_f16_f32(result_vec_4);

        float16x8_t result_half_12 = vcombine_f16(result_half_1, result_half_2);
        float16x8_t result_half_34 = vcombine_f16(result_half_3, result_half_4);

        vst1q_f16(o_block_line + j, result_half_12);
        vst1q_f16(o_block_line + j + 8, result_half_34);
      }

      if (j <= dim_size - 8) {
        float32x4_t vec_acc_o_1 = vld1q_f32(acc_o_line + j);
        float32x4_t vec_acc_o_2 = vld1q_f32(acc_o_line + j + 4);

        float32x4_t result_vec_1 = vmulq_f32(vec_acc_o_1, vec_reciprocal);
        float32x4_t result_vec_2 = vmulq_f32(vec_acc_o_2, vec_reciprocal);

        float16x4_t result_half_1 = vcvt_f16_f32(result_vec_1);
        float16x4_t result_half_2 = vcvt_f16_f32(result_vec_2);

        vst1_f16(o_block_line + j, result_half_1);
        vst1_f16(o_block_line + j + 4, result_half_2);
        j += 8;
      }

      for (; j < dim_size; ++j) { o_block_line[j] = (float16_t)(acc_o_line[j] * reciprocal_logsum); }
    }
  }
};

#endif
#endif

}  // namespace mllm::cpu::fa2