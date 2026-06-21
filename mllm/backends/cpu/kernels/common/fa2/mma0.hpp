// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DataTypes.hpp"
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
template<typename ArchTrait_, typename TileTrait_, typename NumericTrait_, typename LayoutTrait_, typename MemoryTrait_>
struct MMA0 {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  // [Block M, dim_size, Block N]
  //
  // if TileK = -1, then dim_size = dim_size
  // else dim_size = TileK
  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key) {}
};

//===----------------------------------------------------------------------===//
// General Template
//===----------------------------------------------------------------------===//
template<typename ArchTrait_, typename TileTrait_, typename NumericTrait_, typename LayoutTrait_, typename MemoryTrait_>
struct MMA0Tail {
  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  // [Block M, dim_size, Block N]
  //
  // if TileK = -1, then dim_size = dim_size
  // else dim_size = TileK
  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key,
                                    const int32_t real_m_block_size, const int32_t real_n_block_size) {}
};

//===----------------------------------------------------------------------===//
// Native Impl
//===----------------------------------------------------------------------===//
// clang-format off
template<typename ElementAccumulator_, typename ElementCompute_, int M_, int N_>
struct MMA0<
    NativeArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<ElementAccumulator_, ElementCompute_>,
    DefaultMma0Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma0Layout;
  using TileTrait_ = TileTrait<M_, N_, 1>;
  using NumericTrait_ = NumericTrait<ElementAccumulator_, ElementCompute_>;

  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key) {
// C[m][n] += A[m][k] * B[n][k]
#pragma unroll
    for (int m = 0; m < kTileM; ++m) {
#pragma unroll
      for (int n = 0; n < kTileN; ++n) {
        ElementAccumulator sum = 0;
        for (int k = 0; k < dim_size; ++k) {
          sum += static_cast<ElementAccumulator>(query_block[m * stride_query + k] * key_block[n * stride_key + k]);
        }
        acc_s[m * kTileN + n] += sum;
      }
    }
  }
};

// clang-format off
template<typename ElementAccumulator_, typename ElementCompute_, int M_, int N_>
struct MMA0Tail<
    NativeArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<ElementAccumulator_, ElementCompute_>,
    DefaultMma0Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma0Layout;
  using TileTrait_ = TileTrait<M_, N_, 1>;
  using NumericTrait_ = NumericTrait<ElementAccumulator_, ElementCompute_>;

  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key,
                                    const int32_t real_m_block_size, const int32_t real_n_block_size) {
    // C[m][n] += A[m][k] * B[n][k]
    for (int m = 0; m < real_m_block_size; ++m) {
      for (int n = 0; n < real_n_block_size; ++n) {
        ElementAccumulator sum = 0;
        for (int k = 0; k < dim_size; ++k) {
          sum += static_cast<ElementAccumulator>(query_block[m * stride_query + k] * key_block[n * stride_key + k]);
        }
        acc_s[m * kTileN + n] += sum;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#if defined(MLLM_HOST_FEATURE_FP16)
// clang-format off
template<int M_, int N_>
struct MMA0<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<mllm_fp32_t, mllm_fp16_t>,
    DefaultMma0Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma0Layout;
  using TileTrait_ = TileTrait<M_, N_, -1>;
  using NumericTrait_ = NumericTrait<mllm_fp32_t, mllm_fp16_t>;

  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key) {
#pragma unroll
    for (int32_t b_r_idx = 0; b_r_idx < kTileM; ++b_r_idx) {
      const float16_t* q_block_line = query_block + b_r_idx * stride_query;
#pragma unroll
      for (int32_t b_c_idx = 0; b_c_idx < kTileN; ++b_c_idx) {
        const float16_t* k_block_line = key_block + b_c_idx * stride_key;

        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int i = 0;
        // Main loop
        for (; i <= dim_size - 32; i += 32) {
          // Prefetch data
          __builtin_prefetch(q_block_line + i + 64);
          __builtin_prefetch(k_block_line + i + 64);

          // Load data
          float16x8_t q0 = vld1q_f16(q_block_line + i);
          float16x8_t k0 = vld1q_f16(k_block_line + i);
          float16x8_t q1 = vld1q_f16(q_block_line + i + 8);
          float16x8_t k1 = vld1q_f16(k_block_line + i + 8);
          float16x8_t q2 = vld1q_f16(q_block_line + i + 16);
          float16x8_t k2 = vld1q_f16(k_block_line + i + 16);
          float16x8_t q3 = vld1q_f16(q_block_line + i + 24);
          float16x8_t k3 = vld1q_f16(k_block_line + i + 24);

          // MLA
          sum0 = vfmlalq_high_f16(sum0, q0, k0);
          sum0 = vfmlalq_low_f16(sum0, q0, k0);

          sum1 = vfmlalq_high_f16(sum1, q1, k1);
          sum1 = vfmlalq_low_f16(sum1, q1, k1);

          sum2 = vfmlalq_high_f16(sum2, q2, k2);
          sum2 = vfmlalq_low_f16(sum2, q2, k2);

          sum3 = vfmlalq_high_f16(sum3, q3, k3);
          sum3 = vfmlalq_low_f16(sum3, q3, k3);
        }

        // Reduce
        float total = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

        // Loops left
        for (; i <= dim_size - 8; i += 8) {
          float16x8_t q = vld1q_f16(q_block_line + i);
          float16x8_t k = vld1q_f16(k_block_line + i);
          total += vaddvq_f32(vfmlalq_high_f16(vfmlalq_low_f16(vdupq_n_f32(0), q, k), q, k));
        }

        for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

        acc_s[b_r_idx * kTileN + b_c_idx] = total;
      }
    }
  }
};

// clang-format off
template<int M_, int N_>
struct MMA0Tail<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, 1>,
    NumericTrait<mllm_fp32_t, mllm_fp16_t>,
    DefaultMma0Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma0Layout;
  using TileTrait_ = TileTrait<M_, N_, 1>;
  using NumericTrait_ = NumericTrait<mllm_fp32_t, mllm_fp16_t>;

  static constexpr ArchType kArch = ArchTrait_::kArch;

  static constexpr int kTileM = TileTrait_::kTileM;
  static constexpr int kTileN = TileTrait_::kTileN;
  static constexpr int kTileK = TileTrait_::kTileK;

  using ElementAccumulator = NumericTrait_::ElementAccumulator;
  using ElementCompute = NumericTrait_::ElementCompute;

  using LayoutA = LayoutTrait_::LayoutA;
  using LayoutB = LayoutTrait_::LayoutB;
  using LayoutC = LayoutTrait_::LayoutC;

  static constexpr int kAlignment = MemoryTrait_::kAlignment;
  static constexpr int kAlignmentBytes = MemoryTrait_::kAlignment;

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ query_block,
                                    const ElementCompute* __restrict__ key_block, ElementAccumulator* __restrict__ acc_s,
                                    const int32_t dim_size, const int32_t stride_query, const int32_t stride_key,
                                    const int32_t real_m_block_size, const int32_t real_n_block_size) {
    for (int32_t b_r_idx = 0; b_r_idx < real_m_block_size; ++b_r_idx) {
      const float16_t* q_block_line = query_block + b_r_idx * stride_query;
      for (int32_t b_c_idx = 0; b_c_idx < real_n_block_size; ++b_c_idx) {
        const float16_t* k_block_line = key_block + b_c_idx * stride_key;

        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int i = 0;
        // Main loop
        for (; i <= dim_size - 32; i += 32) {
          // Prefetch data
          __builtin_prefetch(q_block_line + i + 64);
          __builtin_prefetch(k_block_line + i + 64);

          // Load data
          float16x8_t q0 = vld1q_f16(q_block_line + i);
          float16x8_t k0 = vld1q_f16(k_block_line + i);
          float16x8_t q1 = vld1q_f16(q_block_line + i + 8);
          float16x8_t k1 = vld1q_f16(k_block_line + i + 8);
          float16x8_t q2 = vld1q_f16(q_block_line + i + 16);
          float16x8_t k2 = vld1q_f16(k_block_line + i + 16);
          float16x8_t q3 = vld1q_f16(q_block_line + i + 24);
          float16x8_t k3 = vld1q_f16(k_block_line + i + 24);

          // MLA
          sum0 = vfmlalq_high_f16(sum0, q0, k0);
          sum0 = vfmlalq_low_f16(sum0, q0, k0);

          sum1 = vfmlalq_high_f16(sum1, q1, k1);
          sum1 = vfmlalq_low_f16(sum1, q1, k1);

          sum2 = vfmlalq_high_f16(sum2, q2, k2);
          sum2 = vfmlalq_low_f16(sum2, q2, k2);

          sum3 = vfmlalq_high_f16(sum3, q3, k3);
          sum3 = vfmlalq_low_f16(sum3, q3, k3);
        }

        // Reduce
        float total = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);

        // Loops left
        for (; i <= dim_size - 8; i += 8) {
          float16x8_t q = vld1q_f16(q_block_line + i);
          float16x8_t k = vld1q_f16(k_block_line + i);
          total += vaddvq_f32(vfmlalq_high_f16(vfmlalq_low_f16(vdupq_n_f32(0), q, k), q, k));
        }

        for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

        acc_s[b_r_idx * kTileN + b_c_idx] = total;
      }
    }
  }
};
#endif  // defined(MLLM_HOST_FEATURE_FP16)
#endif  // defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

}  // namespace mllm::cpu::fa2
