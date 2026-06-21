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
struct InitTemporary {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                    const int32_t dim_size) {}

  static MLLM_FORCE_INLINE void run_decode(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                           const int32_t dim_size) {}
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#if defined(MLLM_HOST_FEATURE_FP16)

// clang-format off
template<int M_, int N_>
struct InitTemporary<
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

  static MLLM_FORCE_INLINE void run(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                    const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < kTileM; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(FA2_FLOAT_NEG_INF);
#pragma unroll
    for (int i = 0; i < kTileM; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < kTileN * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }

  static MLLM_FORCE_INLINE void run_decode(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                           const int32_t dim_size) {
    // Fill 0 to logsum
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
#pragma unroll
    for (int i = 0; i < kTileM; i += 4) { vst1q_f32(logsum + i, zero_vec); }

    // Fill -inf to scoremax
    float32x4_t neg_inf_vec = vdupq_n_f32(FA2_FLOAT_NEG_INF);
#pragma unroll
    for (int i = 0; i < kTileM; i += 4) { vst1q_f32(scoremax + i, neg_inf_vec); }

    // Fill 0 to acc_o
    for (int i = 0; i < dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
  }
};

#endif
#endif

}  // namespace mllm::cpu::fa2
