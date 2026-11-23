// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "tools/hmx_int8_matmul/include/hmx_i8_packed_topk_layout.h"

namespace mllm::cpu::shadownpu {

// Borrowed from a completed head in the FastRPC arena. Only the callback may use it.
struct HmxTopKBitmapView {
  const void* data = nullptr;
  size_t bytes = 0;
};

class HmxTopKBitmapRows {
 public:
  bool reset(HmxTopKBitmapView view, int32_t rows, int32_t valid_columns) noexcept {
    data_ = nullptr;
    if (view.data == nullptr || rows <= 0 || rows > kMaxRows || rows % HMX_I8_TOPK_BITMAP_ROWS != 0
        || valid_columns < rows || valid_columns > UINT16_MAX) {
      return false;
    }
    rows_ = rows;
    prefix_ = valid_columns - rows;
    group_offset_[0] = 0;
    for (int32_t group = 0; group < rows / HMX_I8_TOPK_BITMAP_ROWS; ++group) {
      const int32_t first_visible = prefix_ + group * HMX_I8_TOPK_BITMAP_ROWS + 1;
      group_offset_[group + 1] = group_offset_[group] + hmx_i8_topk_bitmap_group_words(first_visible);
    }
    if (group_offset_[rows / HMX_I8_TOPK_BITMAP_ROWS] > view.bytes / sizeof(uint32_t)) return false;
    data_ = static_cast<const unsigned char*>(view.data);
    return true;
  }

  bool decode(int32_t row, std::span<int32_t> indices, std::span<uint64_t> active_keys = {}) const noexcept {
    if (data_ == nullptr || row < 0 || row >= rows_ || indices.empty()) return false;
    const int32_t visible = prefix_ + row + 1;
    if (indices.size() > static_cast<size_t>(visible)
        || (!active_keys.empty() && active_keys.size() < static_cast<size_t>((visible + 63) / 64))) {
      return false;
    }
    const int32_t group = row / HMX_I8_TOPK_BITMAP_ROWS;
    const int32_t lane = row % HMX_I8_TOPK_BITMAP_ROWS;
    size_t emitted = 0;
    int32_t column = 0;
    const size_t end = group_offset_[group + 1];
    for (size_t word = group_offset_[group] + lane; word < end;
         word += HMX_I8_TOPK_BITMAP_ROWS, column += HMX_I8_TOPK_BITMAP_COLS) {
      uint32_t mask = loadWord(word);
      if (mask == 0) continue;
      if (column + static_cast<int32_t>(std::bit_width(mask)) > visible
          || static_cast<size_t>(std::popcount(mask)) > indices.size() - emitted) {
        return false;
      }
      if (!active_keys.empty()) active_keys[column / 64] |= uint64_t(mask) << (column % 64);
      while (mask != 0) {
        indices[emitted++] = column + std::countr_zero(mask);
        mask &= mask - 1;
      }
    }
    return emitted == indices.size();
  }

  // The producer places each adjacent pair of lanes in the same four-row group.
  // Emit in ascending key order, with independent offsets for each selected row.
  template <class Visitor>
  bool visitPair(int32_t row, size_t first_count, size_t second_count, std::span<uint64_t> active_keys,
                 Visitor&& visit) const noexcept {
    static_assert(HMX_I8_TOPK_BITMAP_ROWS == 4 && HMX_I8_TOPK_BITMAP_COLS == 32);
    if (data_ == nullptr || row < 0 || row >= rows_ - 1 || row % 2 != 0) return false;
    const int32_t visible = prefix_ + row + 1;
    if (first_count == 0 || first_count > static_cast<size_t>(visible)
        || second_count == 0 || second_count > static_cast<size_t>(visible + 1)
        || (!active_keys.empty() && active_keys.size() < static_cast<size_t>((visible + 64) / 64))) {
      return false;
    }
    const int32_t group = row / HMX_I8_TOPK_BITMAP_ROWS;
    const int32_t lane = row % HMX_I8_TOPK_BITMAP_ROWS;
    size_t first = 0, second = 0;
    int32_t column = 0;
    for (size_t word = group_offset_[group] + lane; word < group_offset_[group + 1];
         word += HMX_I8_TOPK_BITMAP_ROWS, column += HMX_I8_TOPK_BITMAP_COLS) {
      const uint32_t a = loadWord(word);
      const uint32_t b = loadWord(word + 1);
      uint32_t pending = a | b;
      if (pending == 0) continue;
      if ((a != 0 && column + static_cast<int32_t>(std::bit_width(a)) > visible)
          || (b != 0 && column + static_cast<int32_t>(std::bit_width(b)) > visible + 1)
          || static_cast<size_t>(std::popcount(a)) > first_count - first
          || static_cast<size_t>(std::popcount(b)) > second_count - second) {
        return false;
      }
      if (!active_keys.empty()) active_keys[column / 64] |= uint64_t(pending) << (column % 64);
      while (pending != 0) {
        const auto bit = std::countr_zero(pending);
        const auto mask = uint32_t{1} << bit;
        const bool in_first = (a & mask) != 0;
        const bool in_second = (b & mask) != 0;
        visit(column + bit, in_first, in_second, first, second);
        first += in_first;
        second += in_second;
        pending &= pending - 1;
      }
    }
    return first == first_count && second == second_count;
  }

  // Rows store probabilities in the same increasing-key order as decode.
  // The consumer can share the same four-row producer words even when its
  // sixteen-row tile starts partway through a producer group.
  template <class Weight, size_t TileRows>
  bool scatterWeights(int32_t first_row, const std::array<size_t, TileRows + 1>& row_offsets,
                      std::span<const Weight> weights, std::span<Weight> panel) const noexcept {
    static_assert(TileRows > 0 && TileRows <= kMaxRows);
    if (data_ == nullptr || first_row < 0 || first_row > rows_ - static_cast<int32_t>(TileRows)
        || row_offsets[0] != 0 || row_offsets[TileRows] != weights.size()
        || panel.size() < static_cast<size_t>(rows_ + prefix_) * TileRows) {
      return false;
    }
    const int32_t last_row = first_row + static_cast<int32_t>(TileRows);
    for (int32_t row = first_row; row < last_row;) {
      const int32_t group = row / HMX_I8_TOPK_BITMAP_ROWS;
      const int32_t group_row = group * HMX_I8_TOPK_BITMAP_ROWS;
      const int32_t end_row = std::min(last_row, group_row + HMX_I8_TOPK_BITMAP_ROWS);
      size_t cursors[HMX_I8_TOPK_BITMAP_ROWS];
      size_t limits[HMX_I8_TOPK_BITMAP_ROWS];
      for (int32_t live = row; live < end_row; ++live) {
        const size_t tile_lane = live - first_row;
        const int32_t lane = live - group_row;
        cursors[lane] = row_offsets[tile_lane];
        limits[lane] = row_offsets[tile_lane + 1];
        if (cursors[lane] >= limits[lane] || limits[lane] > weights.size()) return false;
      }
      int32_t column = 0;
      for (size_t word = group_offset_[group]; word < group_offset_[group + 1];
           word += HMX_I8_TOPK_BITMAP_ROWS, column += HMX_I8_TOPK_BITMAP_COLS) {
        for (int32_t live = row; live < end_row; ++live) {
          const int32_t lane = live - group_row;
          uint32_t mask = loadWord(word + lane);
          if (mask == 0) continue;
          if (column + static_cast<int32_t>(std::bit_width(mask)) > prefix_ + live + 1
              || static_cast<size_t>(std::popcount(mask)) > limits[lane] - cursors[lane]) {
            return false;
          }
          const size_t tile_lane = live - first_row;
          while (mask != 0) {
            const auto key = static_cast<size_t>(column + std::countr_zero(mask));
            panel[key * TileRows + tile_lane] = weights[cursors[lane]++];
            mask &= mask - 1;
          }
        }
      }
      for (int32_t live = row; live < end_row; ++live) {
        const int32_t lane = live - group_row;
        if (cursors[lane] != limits[lane]) return false;
      }
      row = end_row;
    }
    return true;
  }

 private:
  static constexpr int kMaxRows = 512;
  uint32_t loadWord(size_t word) const noexcept {
    uint32_t mask;
    std::memcpy(&mask, data_ + word * sizeof(mask), sizeof(mask));
    return mask;
  }
  const unsigned char* data_ = nullptr;
  int32_t rows_ = 0;
  int32_t prefix_ = 0;
  // reset initializes the complete prefix consumed by decode.
  std::array<size_t, kMaxRows / HMX_I8_TOPK_BITMAP_ROWS + 1> group_offset_;
};

}  // namespace mllm::cpu::shadownpu
