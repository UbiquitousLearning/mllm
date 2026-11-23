// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace shadow_correction_packet {

enum {
  tile_rows = 256,
  in_width = 8960,
  out_width = 1536,
  sparse_capacity = 8192,
  dense_capacity = 2,
  words_per_row = 136,
  packet_words = tile_rows * words_per_row,
  activation_group_width = 64,
  activation_groups_per_row = in_width / activation_group_width,
  activation_groups_per_word = 32,
  activation_mask_words_per_row = (activation_groups_per_row + activation_groups_per_word - 1) / activation_groups_per_word,
};

static_assert(in_width % activation_group_width == 0);

enum { complete = 0, needs_original_tile = 1 };
enum { clip_input = 1, clip_output = 2 };

struct sparse_entry {
  uint32_t channel;
  uint32_t value_bits;
};

struct correction_packet {
  uint32_t status;
  uint32_t sparse_count;
  uint32_t dense_count;
  uint32_t mask_flags;
  uint32_t row_offsets[tile_rows + 1];
  uint32_t dense_row_ids[dense_capacity];
  struct sparse_entry sparse[sparse_capacity];
  uint32_t dense_rows[dense_capacity][in_width];
  uint32_t padding[249];
};

static_assert(sizeof(struct correction_packet) == packet_words * sizeof(uint32_t),
               "QNN tiled Int32 wrapper must have exactly 136 words per sequence row");

}  // namespace shadow_correction_packet
