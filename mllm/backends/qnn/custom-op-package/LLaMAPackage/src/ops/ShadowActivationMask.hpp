// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "ShadowCorrectionPacketKernel.hpp"

#if defined(__hexagon__) || defined(__hexagon)
#include "hvx_internal.h"
#endif

namespace shadow_correction_packet {

#if defined(__hexagon__) || defined(__hexagon)
static inline bool group_can_clip(HVX_Vector first, HVX_Vector second, HVX_Vector signless, HVX_Vector limit) {
  const auto first_exceeds = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(first, signless), limit);
  const auto second_exceeds = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(second, signless), limit);
  const auto either = Q6_Q_or_QQ(first_exceeds, second_exceeds);
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
  return any_predicate_byte(either);
#else
  const auto bytes = Q6_V_vand_QR(either, -1);
  auto words = Q6_V_vor_VV(bytes, Q6_V_vror_VR(bytes, 4));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 8));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 16));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 32));
  words = Q6_V_vor_VV(words, Q6_V_vror_VR(words, 64));
  return Q6_R_vextract_VR(words, 0) != 0;
#endif
}

template<bool aligned>
static inline HVX_Vector read_product_vector(const float* source) {
  if constexpr (aligned) {
    return *reinterpret_cast<const HVX_Vector*>(source);
  } else {
    alignas(HVX_Vector) float lanes[vector_width];
    const volatile float* scalar = source;
    for (unsigned lane = 0; lane < vector_width; ++lane) { lanes[lane] = scalar[lane]; }
    HVX_Vector value;
    memcpy(&value, lanes, sizeof(value));
    return value;
  }
}

template<bool aligned>
static inline void write_product_vector(float* target, HVX_Vector value) {
  if constexpr (aligned) {
    *reinterpret_cast<HVX_Vector*>(target) = value;
  } else {
    alignas(HVX_Vector) float lanes[vector_width];
    memcpy(lanes, &value, sizeof(value));
    volatile float* scalar = target;
    for (unsigned lane = 0; lane < vector_width; ++lane) { scalar[lane] = lanes[lane]; }
  }
}
#endif

// A group is marked conservatively with the smaller of the two sign-specific
// clipping thresholds. Packet assembly still applies the exact signed test.
#if defined(__hexagon__) || defined(__hexagon)
template<bool aligned>
#endif
inline void multiply_mask_tiles(const float* __restrict gate, const float* __restrict up,
                                float* __restrict product, uint32_t* __restrict mask, uint32_t threshold) {
#if defined(__hexagon__) || defined(__hexagon)
  const HVX_Vector signless = Q6_V_vsplat_R(INT32_MAX);
  const HVX_Vector limit = Q6_V_vsplat_R(threshold - 1);
#endif
  for (unsigned row = 0; row < tile_rows; ++row) {
    for (unsigned word = 0; word < activation_mask_words_per_row; ++word) {
      const unsigned first_group = word * activation_groups_per_word;
      const unsigned count = activation_groups_per_row - first_group < activation_groups_per_word
                                 ? activation_groups_per_row - first_group : activation_groups_per_word;
      const unsigned row_channel = row * in_width + first_group * activation_group_width;
#if defined(__hexagon__) || defined(__hexagon)
      if constexpr (aligned) {
        constexpr unsigned vectors_per_fetch = 8192 / sizeof(HVX_Vector);
        constexpr unsigned total_vectors = tile_rows * in_width / vector_width;
        const unsigned next = (row_channel + count * activation_group_width) / vector_width;
        const unsigned remaining = total_vectors - next;
        const unsigned fetch = remaining < vectors_per_fetch ? remaining : vectors_per_fetch;
        if (fetch) {
          l2fetch(gate + next * vector_width, sizeof(HVX_Vector), sizeof(HVX_Vector), fetch, 0);
          l2fetch(up + next * vector_width, sizeof(HVX_Vector), sizeof(HVX_Vector), fetch, 0);
        }
      }
#endif
      uint32_t marked = 0;
      for (unsigned bit = 0; bit < count; ++bit) {
        const unsigned channel = row_channel + bit * activation_group_width;
#if defined(__hexagon__) || defined(__hexagon)
        const auto first = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(read_product_vector<aligned>(gate + channel),
                                                                read_product_vector<aligned>(up + channel)));
        const auto second = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
            read_product_vector<aligned>(gate + channel + vector_width),
            read_product_vector<aligned>(up + channel + vector_width)));
        write_product_vector<aligned>(product + channel, first);
        write_product_vector<aligned>(product + channel + vector_width, second);
        const bool found = group_can_clip(first, second, signless, limit);
#else
        bool found = false;
        for (unsigned lane = channel; lane < channel + activation_group_width; ++lane) {
          product[lane] = gate[lane] * up[lane];
          found |= (word_bits(product + lane) & UINT32_C(0x7FFFFFFF)) >= threshold;
        }
#endif
        marked |= uint32_t{found} << bit;
      }
      mask[row * activation_mask_words_per_row + word] = marked;
    }
  }
}

inline void multiply_with_activation_mask(const float* __restrict gate, const float* __restrict up,
                                          float* __restrict product, uint32_t* __restrict mask,
                                          uint32_t threshold) {
#if defined(__hexagon__) || defined(__hexagon)
  const uintptr_t address_bits =
      reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(product);
  if (address_bits % alignof(HVX_Vector) == 0) {
    return multiply_mask_tiles<true>(gate, up, product, mask, threshold);
  }
  // The ordinary FP32 pointers can be shorter than an aligned vector at the
  // boundary. The rare unaligned path transfers exactly the requested lanes.
  multiply_mask_tiles<false>(gate, up, product, mask, threshold);
#else
  multiply_mask_tiles(gate, up, product, mask, threshold);
#endif
}

}  // namespace shadow_correction_packet
