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
struct MMA1 {
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

  // if TileK = -1, then dim_size = dim_size
  // else dim_size = TileK
  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {}
};

template<typename ArchTrait_, typename TileTrait_, typename NumericTrait_, typename LayoutTrait_, typename MemoryTrait_>
struct MMA1Tail {
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

  // if TileK = -1, then dim_size = dim_size
  // else dim_size = TileK
  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                                    const int32_t real_w_block_size, const int32_t real_v_block_size) {}
};

//===----------------------------------------------------------------------===//
// Native Impl
//===----------------------------------------------------------------------===//
// clang-format off
template<typename ElementAccumulator_, typename ElementCompute_, int M_, int N_>
struct MMA1<
    NativeArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<ElementAccumulator_, ElementCompute_>,
    DefaultMma1Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma1Layout;
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

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
    for (int m_idx = 0; m_idx < kTileM; ++m_idx) {
      for (int n_idx = 0; n_idx < kTileN; ++n_idx) {
        const ElementCompute w = w_block[m_idx * kTileN + n_idx];
        for (int d_idx = 0; d_idx < dim_size; ++d_idx) {
          const ElementCompute v = v_block[n_idx * head_size * dim_size + d_idx];
          acc_o[m_idx * dim_size + d_idx] += static_cast<ElementAccumulator>(w) * static_cast<ElementAccumulator>(v);
        }
      }
    }
  }
};

// clang-format off
template<typename ElementAccumulator_, typename ElementCompute_, int M_, int N_>
struct MMA1Tail<
    NativeArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<ElementAccumulator_, ElementCompute_>,
    DefaultMma1Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = NativeArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma1Layout;
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

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                                    const int32_t real_w_block_size, const int32_t real_v_block_size) {
    for (int m_idx = 0; m_idx < real_w_block_size; ++m_idx) {
      for (int n_idx = 0; n_idx < real_v_block_size; ++n_idx) {
        const ElementCompute w = w_block[m_idx * real_v_block_size + n_idx];
        for (int d_idx = 0; d_idx < dim_size; ++d_idx) {
          const ElementCompute v = v_block[n_idx * head_size * dim_size + d_idx];
          acc_o[m_idx * dim_size + d_idx] += static_cast<ElementAccumulator>(w) * static_cast<ElementAccumulator>(v);
        }
      }
    }
  }
};

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//
// clang-format off
template<int M_, int N_>
struct MMA1<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<float, float16_t>,
    DefaultMma1Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = ArmNeon128ArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma1Layout;
  using TileTrait_ = TileTrait<M_, N_, 1>;
  using NumericTrait_ = NumericTrait<float, float16_t>;

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

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
    float32x4_t acc[kTileM][2];

    for (int d_base = 0; d_base < dim_size; d_base += 8) {
#pragma unroll
      for (int i = 0; i < kTileM; ++i) {
        acc[i][0] = vld1q_f32(acc_o + i * dim_size + d_base);
        acc[i][1] = vld1q_f32(acc_o + i * dim_size + d_base + 4);
      }

      for (int b_c_idx = 0; b_c_idx < kTileN; ++b_c_idx) {
        const float16_t* v_ptr = v_block + b_c_idx * head_size * dim_size + d_base;
        const float16x8_t v = vld1q_f16(v_ptr);

#pragma unroll
        for (int b_r_idx = 0; b_r_idx < kTileM; ++b_r_idx) {
          const float16_t w = w_block[b_r_idx * kTileN + b_c_idx];
          const float16x8_t w_vec = vdupq_n_f16(w);

          acc[b_r_idx][0] = vfmlalq_low_f16(acc[b_r_idx][0], w_vec, v);
          acc[b_r_idx][1] = vfmlalq_high_f16(acc[b_r_idx][1], w_vec, v);
        }
      }

#pragma unroll
      for (int i = 0; i < kTileM; ++i) {
        vst1q_f32(acc_o + i * dim_size + d_base, acc[i][0]);
        vst1q_f32(acc_o + i * dim_size + d_base + 4, acc[i][1]);
      }
    }
  }
};

// clang-format off
template<int M_, int N_>
struct MMA1Tail<
    ArmNeon128ArchTrait,
    TileTrait<M_, N_, -1>,
    NumericTrait<float, float16_t>,
    DefaultMma1Layout,
    MemoryTrait<128>
>
// clang-format on
{
  using ArchTrait_ = ArmNeon128ArchTrait;
  using MemoryTrait_ = MemoryTrait<128>;
  using LayoutTrait_ = DefaultMma1Layout;
  using TileTrait_ = TileTrait<M_, N_, 1>;
  using NumericTrait_ = NumericTrait<float, float16_t>;

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

  static MLLM_FORCE_INLINE void run(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                    ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                                    const int32_t real_w_block_size, const int32_t real_v_block_size) {
    constexpr int kVecLen = 8;

    for (int b_r_idx = 0; b_r_idx < real_w_block_size; ++b_r_idx) {
      int d_base = 0;

      for (; d_base + kVecLen <= real_v_block_size; d_base += kVecLen) {
        float32x4_t acc[2];
        acc[0] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
        acc[1] = vld1q_f32(acc_o + b_r_idx * dim_size + d_base + 4);

        for (int b_c_idx = 0; b_c_idx < kTileN; ++b_c_idx) {
          const float16_t w = w_block[b_r_idx * kTileN + b_c_idx];
          const float16x8_t w_vec = vdupq_n_f16(w);
          const float16_t* v_ptr = v_block + b_c_idx * head_size * dim_size + d_base;
          const float16x8_t v = vld1q_f16(v_ptr);

          acc[0] = vfmlalq_low_f16(acc[0], w_vec, v);
          acc[1] = vfmlalq_high_f16(acc[1], w_vec, v);
        }

        vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc[0]);
        vst1q_f32(acc_o + b_r_idx * dim_size + d_base + 4, acc[1]);
      }

      for (; d_base < real_v_block_size; ++d_base) {
        ElementAccumulator acc_scalar = acc_o[b_r_idx * dim_size + d_base];

        for (int b_c_idx = 0; b_c_idx < kTileN; ++b_c_idx) {
          const ElementCompute w = w_block[b_r_idx * kTileN + b_c_idx];
          const ElementCompute v = v_block[b_c_idx * head_size * dim_size + d_base];
          acc_scalar += static_cast<ElementAccumulator>(w) * static_cast<ElementAccumulator>(v);
        }
        acc_o[b_r_idx * dim_size + d_base] = acc_scalar;
      }
    }
  }
};
#endif

}  // namespace mllm::cpu::fa2
