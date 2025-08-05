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
  MLLM_FORCE_INLINE void operator()(const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
                                    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {}
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
  MLLM_FORCE_INLINE void operator()(const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
                                    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
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

  MLLM_FORCE_INLINE void operator()(const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
                                    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
    // TODO
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

  MLLM_FORCE_INLINE void operator()(const float16_t* __restrict__ w_block, const float16_t* __restrict__ v_block,
                                    float* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                                    const int32_t real_w_block_size, const int32_t real_v_block_size) {
    // TODO
  }
};

//===----------------------------------------------------------------------===//
// Arm Neon 128 Impl
//===----------------------------------------------------------------------===//

}  // namespace mllm::cpu::fa2
