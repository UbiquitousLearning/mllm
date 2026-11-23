// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/SparsePVSme.hpp"

#if defined(MLLM_ANDROID_SME_SPARSE_PV)
#include <arm_sme.h>
#include <arm_sve.h>

#include <cstddef>

namespace mllm::cpu::shadownpu {
namespace {

__attribute__((always_inline)) inline svfloat16_t interleave(
    const __fp16* first, const __fp16* second, svbool_t panel) __arm_streaming {
  return svzip1_f16(svld1_f16(panel, first), svld1_f16(panel, second));
}

template<bool Retire, bool Ordered, class Weight>
__arm_new("za") void accumulateRows(
    const __fp16* values, const int32_t* keys, Weight* panel, int32_t key_count, float* output) __arm_streaming {
  static constexpr __fp16 zero_value[kSparsePvChannels] = {};
  static constexpr __fp16 zero_panel[kSparsePvRows] = {};
  const svbool_t half_panel = svptrue_pat_b16(SV_VL16);
  const svbool_t lanes = svptrue_b16();
  const svbool_t words = svptrue_b32();
  const svfloat32_t empty = svdup_f32(0);
  const auto parity = svand_n_u16_x(lanes, svindex_u16(0, 1), 1);
  const auto even = svcmpeq_n_u16(lanes, parity, 0);
  const auto odd = svcmpeq_n_u16(lanes, parity, 1);

  for (int dim = 0; dim < kSparsePvChannels; dim += 64) {
    svzero_za();
    for (int index = 0; index < key_count; index += 2) {
      const int key0 = keys[index];
      const int key1 = index + 1 < key_count ? keys[index + 1] : -1;
      const __fp16* value0 = values + static_cast<size_t>(key0) * kSparsePvChannels + dim;
      const __fp16* value1 = key1 >= 0 ? values + static_cast<size_t>(key1) * kSparsePvChannels + dim : zero_value + dim;
      auto* panel0 = panel + static_cast<size_t>(key0) * kSparsePvRows;
      const __fp16* panel1 = key1 >= 0 ? panel + static_cast<size_t>(key1) * kSparsePvRows : zero_panel;
      const auto coeff = interleave(panel0, panel1, half_panel);
      if constexpr (Retire) {
        // The second channel tile has loaded the last use of these weights.
        if (dim == kSparsePvChannels - 64) {
          const auto zero = svdup_f16(0);
          svst1_f16(half_panel, panel0, zero);
          if (key1 >= 0) svst1_f16(half_panel, panel + static_cast<size_t>(key1) * kSparsePvRows, zero);
        }
      }
      const auto value_tile0 = interleave(value0, value1, half_panel);
      const auto value_tile1 = interleave(value0 + 16, value1 + 16, half_panel);
      const auto value_tile2 = interleave(value0 + 32, value1 + 32, half_panel);
      const auto value_tile3 = interleave(value0 + 48, value1 + 48, half_panel);
      if constexpr (Ordered) {
        // Preserve FP32 accumulator update order while retaining the interleaved inputs.
        svmopa_za32_f16_m(0, even, even, coeff, value_tile0);
        svmopa_za32_f16_m(1, even, even, coeff, value_tile1);
        svmopa_za32_f16_m(2, even, even, coeff, value_tile2);
        svmopa_za32_f16_m(3, even, even, coeff, value_tile3);
        if (key1 >= 0) {
          svmopa_za32_f16_m(0, odd, odd, coeff, value_tile0);
          svmopa_za32_f16_m(1, odd, odd, coeff, value_tile1);
          svmopa_za32_f16_m(2, odd, odd, coeff, value_tile2);
          svmopa_za32_f16_m(3, odd, odd, coeff, value_tile3);
        }
      } else {
        svmopa_za32_f16_m(0, lanes, lanes, coeff, value_tile0);
        svmopa_za32_f16_m(1, lanes, lanes, coeff, value_tile1);
        svmopa_za32_f16_m(2, lanes, lanes, coeff, value_tile2);
        svmopa_za32_f16_m(3, lanes, lanes, coeff, value_tile3);
      }
    }
    for (unsigned row = 0; row < kSparsePvRows; ++row) {
      float* out = output + row * kSparsePvChannels + dim;
      svst1_f32(words, out, svread_hor_za32_f32_m(empty, words, 0, row));
      svst1_f32(words, out + 16, svread_hor_za32_f32_m(empty, words, 1, row));
      svst1_f32(words, out + 32, svread_hor_za32_f32_m(empty, words, 2, row));
      svst1_f32(words, out + 48, svread_hor_za32_f32_m(empty, words, 3, row));
    }
  }
}

// The source contains 16 FP16 values. LD1H into 32-bit lanes places each
// value in the even halfword and sets the odd halfword to zero.
__attribute__((always_inline)) inline svfloat16_t loadEvenHalves(
    svbool_t words, const __fp16* data) __arm_streaming {
  svuint32_t packed;
  asm("ld1h { %0.s }, %1/z, [%2]" : "=w"(packed) : "Upl"(words), "r"(data), "m"(*reinterpret_cast<const __fp16 (*)[16]>(data)));
  return svreinterpret_f16_u32(packed);
}

// Four ZA tiles hold two 16-row tiles and two 16-channel slices.
__arm_new("za") void accumulateSharedValues(
    const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t count, float* output) __arm_streaming {
  const auto words = svptrue_b32();
  const auto lanes = svptrue_b16();
  const auto parity = svand_n_u16_x(lanes, svindex_u16(0, 1), 1);
  const auto even = svcmpeq_n_u16(lanes, parity, 0);
  const auto empty = svdup_f32(0);

  for (int channel = 0; channel < kSparsePvChannels; channel += 32) {
    svzero_za();
    for (int index = 0; index < count; ++index) {
      const auto encoded = static_cast<uint32_t>(keys[index]);
      const size_t key = encoded & kSparsePvSharedKeyMask;
      const auto* value = values + key * kSparsePvChannels + channel;
      const auto v0 = loadEvenHalves(words, value);
      const auto v1 = loadEvenHalves(words, value + 16);
      const auto* weights = panel + key * kSparsePvSharedRows;
      if (encoded & kSparsePvSharedFirst) {
        const auto coeff = loadEvenHalves(words, weights);
        svmopa_za32_f16_m(0, even, even, coeff, v0);
        svmopa_za32_f16_m(1, even, even, coeff, v1);
      }
      if (encoded & kSparsePvSharedSecond) {
        const auto coeff = loadEvenHalves(words, weights + kSparsePvRows);
        svmopa_za32_f16_m(2, even, even, coeff, v0);
        svmopa_za32_f16_m(3, even, even, coeff, v1);
      }
    }
    for (unsigned row = 0; row < kSparsePvRows; ++row) {
      auto* lo = output + row * kSparsePvChannels + channel;
      auto* hi = lo + kSparsePvRows * kSparsePvChannels;
      svst1_f32(words, lo, svread_hor_za32_f32_m(empty, words, 0, row));
      svst1_f32(words, lo + 16, svread_hor_za32_f32_m(empty, words, 1, row));
      svst1_f32(words, hi, svread_hor_za32_f32_m(empty, words, 2, row));
      svst1_f32(words, hi + 16, svread_hor_za32_f32_m(empty, words, 3, row));
    }
  }
}

}  // namespace

__arm_locally_streaming void sparsePvRowsSme(
    const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count, float* output, bool ordered_keys) noexcept {
  if (ordered_keys) accumulateRows<false, true>(values, keys, panel, key_count, output);
  else accumulateRows<false, false>(values, keys, panel, key_count, output);
}

__arm_locally_streaming void sparsePvRowsSmeRetirePanel(
    const __fp16* values, const int32_t* keys, __fp16* panel, int32_t key_count, float* output, bool ordered_keys) noexcept {
  if (ordered_keys) accumulateRows<true, true>(values, keys, panel, key_count, output);
  else accumulateRows<true, false>(values, keys, panel, key_count, output);
}

__arm_locally_streaming void sparsePvSharedRowsSme(
    const __fp16* values, const int32_t* keys, const __fp16* panel, int32_t key_count, float* output) noexcept {
  accumulateSharedValues(values, keys, panel, key_count, output);
}

}  // namespace mllm::cpu::shadownpu
#endif
