// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mllm/backends/cpu/kernels/common/kai_w4a32_pack.hpp"

#if defined(__aarch64__)
#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#endif

namespace {

constexpr auto kTileName = "qai8dxp1x8_qsi4c32p8x8_1x8x32";

TEST(KaiW4A32PackTest, PacksDeterministicallyOnHost) {
  constexpr int kOutChannels = 17;
  constexpr int kInChannels = 64;

  std::vector<float> weights(kOutChannels * kInChannels);
  for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
    weights[index] =
        std::sin(static_cast<float>(index) * 0.071F) * 0.30F - std::cos(static_cast<float>(index) * 0.019F) * 0.05F;
  }

  const auto packed_size = mllm::cpu::common::kaiW4A32PackedSize(kOutChannels, kInChannels, kTileName);
  EXPECT_EQ(packed_size, 1056);

  std::vector<std::uint8_t> first(packed_size, 0xA5);
  std::vector<std::uint8_t> second(packed_size, 0x5A);
  mllm::cpu::common::kaiW4A32QuantizeAndPack(first.data(), weights.data(), nullptr, kOutChannels, kInChannels, kTileName);
  mllm::cpu::common::kaiW4A32QuantizeAndPack(second.data(), weights.data(), nullptr, kOutChannels, kInChannels, kTileName);

  EXPECT_EQ(first, second);
  EXPECT_TRUE(std::any_of(first.begin(), first.end(), [](std::uint8_t value) { return value != 0; }));
}

TEST(KaiW4A32PackTest, RejectsInvalidDimensionsAndTile) {
  EXPECT_THROW(mllm::cpu::common::kaiW4A32PackedSize(17, 63, kTileName), std::invalid_argument);
  EXPECT_THROW(mllm::cpu::common::kaiW4A32PackedSize(17, 64, "unsupported"), std::invalid_argument);
}

#if defined(__aarch64__)
TEST(KaiW4A32PackTest, SharedTileParametersMatchArmUkernels) {
  using Getter = std::size_t (*)();
  struct TileCase {
    std::string_view name;
    Getter get_nr;
    Getter get_kr;
    Getter get_sr;
  };

  const std::array<TileCase, 6> cases = {{
      {"qai8dxp1x8_qsi4c32p4x8_1x4x32", kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
       kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
       kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod},
      {"qai8dxp1x8_qsi4c32p8x8_1x8x32", kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
       kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
       kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod},
      {"qai8dxp4x8_qsi4c32p4x8_8x4x32", kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
       kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
       kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm},
      {"qai8dxp4x8_qsi4c32p4x8_16x4x32", kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
       kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
       kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm},
      {"qai8dxp4x8_qsi4c32p8x8_4x8x32", kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
       kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
       kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm},
      {"qai8dxp1x4_qsi4c32p4x4_1x4", kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
       kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
       kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod},
  }};

  for (const auto& test_case : cases) {
    const auto tile = mllm::cpu::common::kaiW4A32TileFromName(test_case.name);
    EXPECT_EQ(tile.nr, test_case.get_nr()) << test_case.name;
    EXPECT_EQ(tile.kr, test_case.get_kr()) << test_case.name;
    EXPECT_EQ(tile.sr, test_case.get_sr()) << test_case.name;
  }
}

TEST(KaiW4A32PackTest, BatchedRowsMatchIndependentRows) {
  constexpr int kRows = 7;
  constexpr int kOutChannels = 17;
  constexpr int kInChannels = 64;
  using Helper = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk;
  constexpr auto kTile = Helper::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;

  std::vector<float> weights(kOutChannels * kInChannels);
  std::vector<float> input(kRows * kInChannels);
  for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
    weights[index] = std::sin(static_cast<float>(index) * 0.071F) * 0.30F
                     - std::cos(static_cast<float>(index) * 0.019F) * 0.05F;
  }
  for (int index = 0; index < static_cast<int>(input.size()); ++index) {
    input[index] = std::sin(static_cast<float>(index) * 0.037F) * 0.50F
                   + std::cos(static_cast<float>(index) * 0.013F) * 0.07F;
  }

  Helper helper;
  std::vector<std::uint8_t> packed(helper.quant_pack_rhs_size(kOutChannels, kInChannels, kTile));
  helper.quant_pack_rhs_offline(packed.data(), weights.data(), nullptr, kOutChannels, kInChannels, kTile);
  std::vector<std::uint8_t> batched_workspace(helper.workspace_size(kRows, kInChannels, kTile));
  std::vector<float> batched(kRows * kOutChannels);
  helper.matmul(batched.data(), input.data(), packed.data(), batched_workspace.data(), kRows, kInChannels, kOutChannels,
                kTile, 1);

  std::vector<std::uint8_t> row_workspace(helper.workspace_size(1, kInChannels, kTile));
  std::vector<float> row(kOutChannels);
  for (int row_index = 0; row_index < kRows; ++row_index) {
    helper.matmul(row.data(), input.data() + row_index * kInChannels, packed.data(), row_workspace.data(), 1, kInChannels,
                  kOutChannels, kTile, 1);
    for (int channel = 0; channel < kOutChannels; ++channel) {
      EXPECT_FLOAT_EQ(batched[row_index * kOutChannels + channel], row[channel])
          << "row=" << row_index << " channel=" << channel;
    }
  }
}
#endif

}  // namespace
