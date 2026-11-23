#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "ShadowCorrectionPacketLayout.hpp"
#include <string.h>

#if defined(__hexagon__) || defined(__hexagon)
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif

namespace shadow_correction_packet {

constexpr int vector_width = 32;

enum class code_encoding : uint16_t { signed_fixed = 0, offset_binary = 0x8000 };

static inline bool valid_controls(uint32_t positive, uint32_t negative, uint32_t flags) {
  return (flags & ~(clip_input | clip_output)) == 0 && positive != 0 && negative != 0
         && positive < UINT32_C(0x7F800000) && negative < UINT32_C(0x7F800000);
}

static inline uint32_t word_bits(const float *value) {
  uint32_t bits;
  memcpy(&bits, value, sizeof(bits));
  return bits;
}

#if defined(__hexagon__) || defined(__hexagon)
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
static inline bool any_predicate_byte(HVX_VectorPred predicate) {
  // The last word is the inclusive count of all active predicate bytes.
  constexpr int last_word_offset = sizeof(HVX_Vector) - sizeof(uint32_t);
  return Q6_R_vextract_VR(Q6_Vw_prefixsum_Q(predicate), last_word_offset) != 0;
}
#endif

#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
static inline bool activation_group_can_clip(const uint16_t* codes) {
  HVX_Vector values;
  memcpy(&values, codes, sizeof(values));
  constexpr uint16_t lower_last = 1;
  constexpr uint16_t upper_first = UINT16_MAX - 1;
  const auto near_low = Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R(lower_last + 1), values);
  const auto near_high = Q6_Q_vcmp_gt_VuhVuh(values, Q6_Vh_vsplat_R(upper_first - 1));
  return any_predicate_byte(Q6_Q_or_QQ(near_low, near_high));
}
#endif

static inline uint32_t clipping_mask(const float *values, uint32_t positive, uint32_t negative) {
  // One uniquely weighted bit per word reduces the predicate to a scalar lane mask.
  static const uint32_t lane_bits[vector_width] __attribute__((aligned(128))) = {
      1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7,
      1u << 8, 1u << 9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
      1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
      1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
  };
  HVX_Vector bits, lane_values;
  memcpy(&bits, values, sizeof(bits));
  const HVX_Vector zero = Q6_V_vsplat_R(0);
  const HVX_Vector magnitude = Q6_V_vand_VV(bits, Q6_V_vsplat_R(INT32_MAX));
  const HVX_VectorPred signs = Q6_Q_vcmp_gt_VwVw(zero, bits);
  const HVX_Vector cuts = Q6_V_vmux_QVV(signs, Q6_V_vsplat_R(negative - 1), Q6_V_vsplat_R(positive - 1));
  const HVX_VectorPred exceed = Q6_Q_vcmp_gt_VwVw(magnitude, cuts);
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
  // Most activation vectors contain no clipping entries. Avoid forming the
  // individual lane bits when the vector contributes nothing to the packet.
  if (!any_predicate_byte(exceed)) return 0;
#endif
  memcpy(&lane_values, lane_bits, sizeof(lane_values));
  HVX_Vector selected = Q6_V_vmux_QVV(exceed, lane_values, zero);
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 4));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 8));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 16));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 32));
  selected = Q6_V_vor_VV(selected, Q6_V_vror_VR(selected, 64));
  return (uint32_t)Q6_R_vextract_VR(selected, 0);
}
#else
static inline uint32_t clipping_mask(const float *values, uint32_t positive, uint32_t negative) {
  uint32_t selected = 0;
  for (unsigned lane = 0; lane < vector_width; ++lane) {
    const uint32_t bits = word_bits(values + lane);
    const uint32_t cut = bits >> 31 ? negative : positive;
    if ((bits & UINT32_C(0x7FFFFFFF)) >= cut) {
      selected |= UINT32_C(1) << lane;
    }
  }
  return selected;
}
#endif

static int row_needs_output_repair(const int16_t *codes, uint16_t minimum, uint16_t maximum) {
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
  constexpr unsigned codes_per_vector = sizeof(HVX_Vector) / sizeof(int16_t);
  static_assert(out_width % codes_per_vector == 0);
  const HVX_Vector min_codes = Q6_Vh_vsplat_R(minimum);
  const HVX_Vector max_codes = Q6_Vh_vsplat_R(maximum);
  const HVX_Vector zero = Q6_V_vsplat_R(0);
  HVX_VectorPred found = Q6_Q_vcmp_gt_VwVw(zero, zero);
  for (unsigned channel = 0; channel < out_width; channel += codes_per_vector) {
    HVX_Vector chunk;
    memcpy(&chunk, codes + channel, sizeof(chunk));
    const auto at_limit = Q6_Q_or_QQ(Q6_Q_vcmp_eq_VhVh(chunk, min_codes),
                                    Q6_Q_vcmp_eq_VhVh(chunk, max_codes));
    found = Q6_Q_or_QQ(found, at_limit);
  }
  return any_predicate_byte(found);
#else
  const auto *words = reinterpret_cast<const uint16_t *>(codes);
  for (unsigned channel = 0; channel < out_width; ++channel) {
    if (words[channel] == minimum || words[channel] == maximum) {
      return 1;
    }
  }
  return 0;
#endif
}

static inline bool append_clipped_vector(correction_packet& packet, const float* values, unsigned channel,
                                         uint32_t positive, uint32_t negative, unsigned& sparse_count) {
  uint32_t lanes = clipping_mask(values + channel, positive, negative);
  if (static_cast<unsigned>(__builtin_popcount(lanes)) > sparse_capacity - sparse_count) { return false; }
  while (lanes) {
    const unsigned column = channel + static_cast<unsigned>(__builtin_ctz(lanes));
    const uint32_t bits = word_bits(values + column);
    if ((bits & UINT32_C(0x7FFFFFFF)) >= UINT32_C(0x7F800000)) { return false; }
    packet.sparse[sparse_count++] = sparse_entry{column, bits};
    lanes &= lanes - 1;
  }
  return true;
}

template<bool use_activation_codes = false, bool use_activation_mask = false>
inline int pack_shadow_correction_tile(const float *activation, const int16_t *codes,
                                uint32_t positive, uint32_t negative, uint32_t flags,
                                struct correction_packet *packet,
                                code_encoding encoding = code_encoding::signed_fixed,
                                const uint16_t* activation_codes = nullptr,
                                const uint32_t* activation_mask = nullptr) {
  static_assert(!(use_activation_codes && use_activation_mask));
  static_assert(activation_group_width == 2 * vector_width);
  memset(packet, 0, sizeof(*packet));
  packet->status = needs_original_tile;
  packet->mask_flags = flags;
  if (!valid_controls(positive, negative, flags)) { return needs_original_tile; }

  // Adding the offset 32768 to a signed code flips only its sign bit.
  const uint16_t bias = static_cast<uint16_t>(encoding);
  const uint16_t minimum = static_cast<uint16_t>(INT16_MIN) ^ bias;
  const uint16_t maximum = static_cast<uint16_t>(INT16_MAX) ^ bias;
  unsigned sparse_count = 0;
  unsigned dense_count = 0;
  for (unsigned row = 0; row < tile_rows; ++row) {
    const float *values = activation + row * in_width;
    if (!(flags & clip_output) && row_needs_output_repair(codes + row * out_width, minimum, maximum)) {
      if (dense_count == dense_capacity) return needs_original_tile;
      packet->dense_row_ids[dense_count] = row;
      memcpy(packet->dense_rows[dense_count], values, in_width * sizeof(float));
      ++dense_count;
    } else if (!(flags & clip_input)) {
      if constexpr (use_activation_mask) {
        const auto* row_mask = activation_mask + row * activation_mask_words_per_row;
        for (unsigned word = 0; word < activation_mask_words_per_row; ++word) {
          uint32_t groups = row_mask[word];
          while (groups) {
            const unsigned group = word * activation_groups_per_word + static_cast<unsigned>(__builtin_ctz(groups));
            if (group >= activation_groups_per_row) { return needs_original_tile; }
            const unsigned channel = group * activation_group_width;
            if (!append_clipped_vector(*packet, values, channel, positive, negative, sparse_count)
                || !append_clipped_vector(*packet, values, channel + vector_width, positive, negative, sparse_count)) {
              return needs_original_tile;
            }
            groups &= groups - 1;
          }
        }
      } else {
        for (unsigned channel = 0; channel < in_width; channel += vector_width) {
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 81
          if constexpr (use_activation_codes) {
            static_assert(activation_group_width == sizeof(HVX_Vector) / sizeof(uint16_t));
            if (channel % activation_group_width == 0
                && !activation_group_can_clip(activation_codes + row * in_width + channel)) {
              channel += vector_width;
              continue;
            }
          }
#endif
          if (!append_clipped_vector(*packet, values, channel, positive, negative, sparse_count)) {
            return needs_original_tile;
          }
        }
      }
    }
    packet->row_offsets[row + 1] = sparse_count;
  }

  packet->sparse_count = sparse_count;
  packet->dense_count = dense_count;
  packet->status = complete;
  return complete;
}

}  // namespace shadow_correction_packet
