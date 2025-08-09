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
struct Rescale {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                    const int32_t dim_size) {}

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                         const int32_t dim_size) {}
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#if defined(MLLM_HOST_FEATURE_FP16)

// clang-format off
template<int M_, int N_>
struct Rescale<
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

  static MLLM_FORCE_INLINE void run(ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                    const int32_t dim_size) {
#pragma unroll
    for (int i = 0; i < kTileM; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      int32_t j = 0;
      for (; j + 15 < dim_size; j += 16) {
        float32x4_t acc0 = vld1q_f32(row_ptr + j + 0);
        float32x4_t acc1 = vld1q_f32(row_ptr + j + 4);
        float32x4_t acc2 = vld1q_f32(row_ptr + j + 8);
        float32x4_t acc3 = vld1q_f32(row_ptr + j + 12);

        acc0 = vmulq_f32(acc0, scale_v);
        acc1 = vmulq_f32(acc1, scale_v);
        acc2 = vmulq_f32(acc2, scale_v);
        acc3 = vmulq_f32(acc3, scale_v);

        vst1q_f32(row_ptr + j + 0, acc0);
        vst1q_f32(row_ptr + j + 4, acc1);
        vst1q_f32(row_ptr + j + 8, acc2);
        vst1q_f32(row_ptr + j + 12, acc3);
      }
      for (; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }

  static MLLM_FORCE_INLINE void run_tail(const int32_t real_m_block_size, const int32_t real_n_block_size,
                                         ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                         const int32_t dim_size) {
    for (int i = 0; i < real_m_block_size; ++i) {
      const float scale = score_scale[i];
      const float32x4_t scale_v = vdupq_n_f32(scale);
      float* row_ptr = acc_o + i * dim_size;
      int32_t j = 0;
      for (; j + 15 < dim_size; j += 16) {
        float32x4_t acc0 = vld1q_f32(row_ptr + j + 0);
        float32x4_t acc1 = vld1q_f32(row_ptr + j + 4);
        float32x4_t acc2 = vld1q_f32(row_ptr + j + 8);
        float32x4_t acc3 = vld1q_f32(row_ptr + j + 12);

        acc0 = vmulq_f32(acc0, scale_v);
        acc1 = vmulq_f32(acc1, scale_v);
        acc2 = vmulq_f32(acc2, scale_v);
        acc3 = vmulq_f32(acc3, scale_v);

        vst1q_f32(row_ptr + j + 0, acc0);
        vst1q_f32(row_ptr + j + 4, acc1);
        vst1q_f32(row_ptr + j + 8, acc2);
        vst1q_f32(row_ptr + j + 12, acc3);
      }
      for (; j < dim_size; j += 4) {
        float32x4_t acc = vld1q_f32(row_ptr + j);
        acc = vmulq_f32(acc, scale_v);
        vst1q_f32(row_ptr + j, acc);
      }
    }
  }
};

#endif
#endif

}  // namespace mllm::cpu::fa2