// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu::fa2 {

enum class ArchType {
  kNative,
  kArmNeon128,
  kArmSme,
  kX86Avx512,
  kX86Avx2,
  kX86Avx,
};

template<typename ArchTag_>
struct ArchTrait {
  using ArchTag = ArchTag_;
};

struct DefaultArchTrait : public ArchTrait<DefaultArchTrait> {
  static constexpr ArchType kArch = ArchType::kNative;
};

struct NativeArchTrait : public ArchTrait<NativeArchTrait> {
  static constexpr ArchType kArch = ArchType::kNative;
};

struct ArmNeon128ArchTrait : public ArchTrait<ArmNeon128ArchTrait> {
  static constexpr ArchType kArch = ArchType::kArmNeon128;
};

struct ArmX86Avx512ArchTrait : public ArchTrait<ArmX86Avx512ArchTrait> {
  static constexpr ArchType kArch = ArchType::kX86Avx512;
};

struct ArmX86Avx2ArchTrait : public ArchTrait<ArmX86Avx2ArchTrait> {
  static constexpr ArchType kArch = ArchType::kX86Avx2;
};

struct ArmX86AvxArchTrait : public ArchTrait<ArmX86AvxArchTrait> {
  static constexpr ArchType kArch = ArchType::kX86Avx;
};

struct RowMajorLayout {
  static constexpr bool kIsRowMajor = true;
  static constexpr bool kIsColMajor = false;
};

struct ColumnMajorLayout {
  static constexpr bool kIsRowMajor = false;
  static constexpr bool kIsColMajor = true;
};

template<typename LayoutA_, typename LayoutB_, typename LayoutC_>
struct MmaLayout {
  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
  using LayoutC = LayoutC_;
};

// mma0 (Q * K^T)
using DefaultMma0Layout = MmaLayout<RowMajorLayout, ColumnMajorLayout, RowMajorLayout>;

// mma1 (S' * V)
using DefaultMma1Layout = MmaLayout<RowMajorLayout, RowMajorLayout, RowMajorLayout>;

template<typename ElementAccumulator_, typename ElementCompute_>
struct NumericTrait {
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
};

struct DefaultNumericTrait : public NumericTrait<mllm_fp32_t, mllm_fp16_t> {};

template<int kAlignment_ = 128>
struct MemoryTrait {
  static constexpr int kAlignment = kAlignment_;
  static constexpr int kAlignmentBytes = kAlignment_ / 8;
};

template<int kTileM_ = 4, int kTileN_ = 4, int kTileK_ = -1>
struct TileTrait {
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
};

template<typename Arch_ = DefaultArchTrait, typename Mma0Layout_ = DefaultMma0Layout, typename Mma1Layout_ = DefaultMma1Layout,
         typename Numeric_ = DefaultNumericTrait, typename Memory_ = MemoryTrait<>, typename Tile_ = TileTrait<>>
struct KernelConfig {
  using Arch = Arch_;
  using Mma0Layout = Mma0Layout_;
  using Mma1Layout = Mma1Layout_;
  using Numeric = Numeric_;
  using Memory = Memory_;
  using Tile = Tile_;
};

template<typename KernelConfig_, bool kNormalCausalMask_, bool kSlidingWindow_, bool kDropout_, bool kHighPrecision_ = false>
struct FlashAttention2Config {
  using KernelConfig = KernelConfig_;

  using Arch = typename KernelConfig_::Arch;
  using Mma0Layout = typename KernelConfig_::Mma0Layout;
  using Mma1Layout = typename KernelConfig_::Mma1Layout;
  using Numeric = typename KernelConfig_::Numeric;
  using Memory = typename KernelConfig_::Memory;
  using Tile = typename KernelConfig_::Tile;

  static constexpr bool kHasCausalMask = kNormalCausalMask_;
  static constexpr bool kHasSlidingWindow = kSlidingWindow_;
  static constexpr bool kHasDropout = kDropout_;
  static constexpr bool kHighPrecision = kHighPrecision_;
};

using DefaultFlashAttention2Config =
    FlashAttention2Config<KernelConfig<DefaultArchTrait, DefaultMma0Layout, DefaultMma1Layout, DefaultNumericTrait,
                                       MemoryTrait<128>, TileTrait<4, 4, -1>>,
                          true,   // kHasCausalMask
                          false,  // kHasSlidingWindow
                          false,  // kHasDropout
                          false   //  kHighPrecision_
                          >;

#define FA2_FLOAT_NEG_INF std::numeric_limits<float>::lowest()

}  // namespace mllm::cpu::fa2
