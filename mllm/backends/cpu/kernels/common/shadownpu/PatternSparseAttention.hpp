// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>

#include "mllm/backends/cpu/kernels/arm/SparsePVSme.hpp"
#include "mllm/backends/cpu/kernels/arm/softmax.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/HmxTopKBitmap.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/vec_dot.hpp"
#include "mllm/backends/cpu/kernels/x86/softmax.hpp"

namespace mllm::cpu::shadownpu {

inline void buildRandomPatternIndices(int32_t valid, int32_t keep, uint64_t seed, int32_t* indices, uint64_t* random_bits) {
  if (keep == valid) {
    for (int32_t index = 0; index < valid; ++index) { indices[index] = index; }
    return;
  }

  uint32_t random = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(seed >> 32) ^ 0x6d2b79f5U;
  if (random == 0) { random = 0x9e3779b9U; }
  const auto next_random = [&random]() {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    return random;
  };
  const auto uniform_bounded = [&next_random](uint32_t bound) {
    while (true) {
      const uint32_t value = next_random();
      const uint64_t product = static_cast<uint64_t>(value) * bound;
      const uint32_t low = static_cast<uint32_t>(product);
      if (low < bound) {
        const uint32_t threshold = static_cast<uint32_t>(-bound) % bound;
        if (low < threshold) { continue; }
      }
      return static_cast<uint32_t>(product >> 32);
    }
  };

  for (int32_t upper = valid - keep; upper < valid; ++upper) {
    const int32_t candidate = static_cast<int32_t>(uniform_bounded(static_cast<uint32_t>(upper + 1)));
    const uint64_t candidate_mask = uint64_t{1} << (candidate & 63);
    const bool already_selected = (random_bits[candidate >> 6] & candidate_mask) != 0;
    const int32_t selected = already_selected ? upper : candidate;
    random_bits[selected >> 6] |= uint64_t{1} << (selected & 63);
  }

  int32_t output_index = 0;
  for (int32_t word_index = 0; word_index < (valid + 63) / 64; ++word_index) {
    uint64_t word = random_bits[word_index];
    random_bits[word_index] = 0;
    while (word != 0) {
      const int32_t bit = static_cast<int32_t>(__builtin_ctzll(word));
      indices[output_index++] = word_index * 64 + bit;
      word &= word - 1;
    }
  }
}

inline void buildRandomPatternIndices(int32_t valid, int32_t keep, uint64_t seed, int32_t* indices) {
  std::vector<uint64_t> random_bits(static_cast<size_t>((valid + 63) / 64), 0);
  buildRandomPatternIndices(valid, keep, seed, indices, random_bits.data());
}

inline uint64_t makeRandomRowSeed(uint32_t seed, int32_t absolute_query, int32_t batch, int32_t head) {
  uint64_t mixed = (static_cast<uint64_t>(seed) << 32)
                   ^ (static_cast<uint64_t>(static_cast<uint32_t>(absolute_query + 1)) * 0x9e3779b97f4a7c15ULL)
                   ^ (static_cast<uint64_t>(static_cast<uint32_t>(batch + 1)) * 0xbf58476d1ce4e5b9ULL)
                   ^ (static_cast<uint64_t>(static_cast<uint32_t>(head + 1)) * 0x94d049bb133111ebULL);
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
  return mixed ^ (mixed >> 31);
}

inline void buildStructuredPatternIndices(int32_t valid, int32_t keep, float local_ratio, int32_t prefix_tokens,
                                          int32_t* indices) {
  const int32_t local_keep = std::clamp(static_cast<int32_t>(std::ceil(local_ratio * static_cast<float>(keep))), 0, keep);
  const int32_t history_end = valid - local_keep;
  const int32_t nonlocal_keep = keep - local_keep;
  const int32_t prefix_keep = std::min(prefix_tokens, std::min(nonlocal_keep, history_end));
  for (int32_t index = 0; index < prefix_keep; ++index) { indices[index] = index; }

  const int32_t uniform_keep = nonlocal_keep - prefix_keep;
  const int32_t uniform_begin = prefix_keep;
  const int32_t uniform_len = history_end - uniform_begin;
  if (uniform_keep == 1) {
    indices[prefix_keep] = uniform_begin;
  } else if (uniform_keep > 1) {
    for (int32_t index = 0; index < uniform_keep; ++index) {
      indices[prefix_keep + index] =
          static_cast<int32_t>(uniform_begin + static_cast<int64_t>(index) * (uniform_len - 1) / (uniform_keep - 1));
    }
  }
  for (int32_t index = 0; index < local_keep; ++index) { indices[nonlocal_keep + index] = history_end + index; }
}

struct PatternSparseAttentionF32Params {
  int32_t batch = 0;
  int32_t query_heads = 0;
  int32_t kv_heads = 0;
  int32_t query_len = 0;
  int32_t key_len = 0;
  int32_t query_key_dim = 0;
  int32_t value_dim = 0;
  std::array<int32_t, 4> query_stride{};
  std::array<int32_t, 4> key_stride{};
  std::array<int32_t, 4> value_stride{};
  std::array<int32_t, 4> output_stride{};
  float sparsity = 0.0F;
  bool causal_mask = true;
  float local_ratio = 0.5F;
  int32_t prefix_tokens = 0;
  int32_t dense_tokens = 0;
  bool random_pattern = false;
  uint32_t random_seed = 1;
  const float* head_retentions = nullptr;
};

template<typename KeyType, typename ValueType>
inline bool patternSparseAttentionRangeTyped(const PatternSparseAttentionF32Params& params, const float* query,
                                             const KeyType* key, const ValueType* value, float* output, size_t matrix_begin,
                                             size_t matrix_count, int32_t query_row_begin, int32_t query_row_count,
                                             const int32_t* external_indices = nullptr, size_t external_head_stride = 0,
                                             const int32_t* external_row_offsets = nullptr,
                                             size_t external_row_offsets_head_stride = 0) {
  if (query == nullptr || key == nullptr || value == nullptr || output == nullptr || params.batch <= 0
      || params.query_heads <= 0 || params.kv_heads <= 0 || params.query_heads % params.kv_heads != 0 || params.query_len <= 0
      || params.key_len < params.query_len || params.query_key_dim <= 0 || params.value_dim <= 0 || params.sparsity < 0.0F
      || params.sparsity >= 1.0F || params.local_ratio < 0.0F || params.local_ratio > 1.0F || params.prefix_tokens < 0
      || params.dense_tokens < 0) {
    return false;
  }

  if ((external_indices == nullptr) != (external_row_offsets == nullptr)) { return false; }
  if (external_indices != nullptr && (external_head_stride == 0 || external_row_offsets_head_stride == 0)) { return false; }
  const size_t total_matrices = static_cast<size_t>(params.batch) * params.query_heads;
  if (matrix_begin > total_matrices || matrix_count > total_matrices - matrix_begin || query_row_begin < 0
      || query_row_begin > params.query_len || query_row_count < 0 || query_row_count > params.query_len - query_row_begin) {
    return false;
  }
  if (matrix_count == 0 || query_row_count == 0) { return true; }

  const int32_t groups = params.query_heads / params.kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(params.query_key_dim));
  std::vector<int32_t> selected_scratch(static_cast<size_t>(params.key_len));
  std::vector<float> probabilities(static_cast<size_t>(params.key_len));
  std::vector<uint64_t> random_bits(static_cast<size_t>((params.key_len + 63) / 64), 0);

  for (size_t matrix = matrix_begin; matrix < matrix_begin + matrix_count; ++matrix) {
    const int32_t batch = static_cast<int32_t>(matrix / static_cast<size_t>(params.query_heads));
    const int32_t query_head = static_cast<int32_t>(matrix % static_cast<size_t>(params.query_heads));
    const int32_t kv_head = query_head / groups;
    const float density = params.head_retentions == nullptr ? 1.0F - params.sparsity : params.head_retentions[query_head];
    if (density < 0.0F || density > 1.0F) { return false; }
    for (int32_t query_index = query_row_begin; query_index < query_row_begin + query_row_count; ++query_index) {
      const int32_t valid = params.causal_mask && params.query_len > 1
                                ? std::min(params.key_len, params.key_len - params.query_len + query_index + 1)
                                : params.key_len;
      const int32_t expected_keep =
          valid <= params.dense_tokens
              ? valid
              : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
      const int32_t* selected = nullptr;
      int32_t keep = expected_keep;
      if (external_indices != nullptr) {
        const int32_t* row_offsets = external_row_offsets + static_cast<size_t>(query_head) * external_row_offsets_head_stride;
        const int32_t begin = row_offsets[query_index];
        const int32_t end = row_offsets[query_index + 1];
        if (begin < 0 || end <= begin || end - begin != expected_keep || static_cast<size_t>(end) > external_head_stride) {
          return false;
        }
        keep = end - begin;
        selected = external_indices + matrix * external_head_stride + begin;
      } else if (params.random_pattern) {
        buildRandomPatternIndices(
            valid, keep,
            makeRandomRowSeed(params.random_seed, params.key_len - params.query_len + query_index, batch, query_head),
            selected_scratch.data(), random_bits.data());
        selected = selected_scratch.data();
      } else {
        buildStructuredPatternIndices(valid, keep, params.local_ratio, params.prefix_tokens, selected_scratch.data());
        selected = selected_scratch.data();
      }

      const float* query_row = query + static_cast<size_t>(batch) * params.query_stride[0]
                               + static_cast<size_t>(query_head) * params.query_stride[1]
                               + static_cast<size_t>(query_index) * params.query_stride[2];
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
        const int32_t key_index = selected[selected_index];
        if (key_index < 0 || key_index >= valid) { return false; }
        const KeyType* key_row = key + static_cast<size_t>(batch) * params.key_stride[0]
                                 + static_cast<size_t>(kv_head) * params.key_stride[1]
                                 + static_cast<size_t>(key_index) * params.key_stride[2];
        float score = 0.0F;
        for (int32_t dim = 0; dim < params.query_key_dim; ++dim) {
          score += query_row[static_cast<size_t>(dim) * params.query_stride[3]]
                   * static_cast<float>(key_row[static_cast<size_t>(dim) * params.key_stride[3]]);
        }
        probabilities[selected_index] = score * scale;
        maximum = std::max(maximum, probabilities[selected_index]);
      }

      float denominator = 0.0F;
      for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
        probabilities[selected_index] = std::exp(probabilities[selected_index] - maximum);
        denominator += probabilities[selected_index];
      }
      const float inverse_denominator = 1.0F / denominator;
      float* output_row = output + static_cast<size_t>(batch) * params.output_stride[0]
                          + static_cast<size_t>(query_head) * params.output_stride[1]
                          + static_cast<size_t>(query_index) * params.output_stride[2];
      for (int32_t dim = 0; dim < params.value_dim; ++dim) {
        float result = 0.0F;
        for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
          const int32_t key_index = selected[selected_index];
          const ValueType* value_row = value + static_cast<size_t>(batch) * params.value_stride[0]
                                       + static_cast<size_t>(kv_head) * params.value_stride[1]
                                       + static_cast<size_t>(key_index) * params.value_stride[2];
          result += probabilities[selected_index] * inverse_denominator
                    * static_cast<float>(value_row[static_cast<size_t>(dim) * params.value_stride[3]]);
        }
        output_row[static_cast<size_t>(dim) * params.output_stride[3]] = result;
      }
    }
  }
  return true;
}

template<typename KeyType, typename ValueType>
inline bool patternSparseAttentionTyped(const PatternSparseAttentionF32Params& params, const float* query, const KeyType* key,
                                        const ValueType* value, float* output, const int32_t* external_indices = nullptr,
                                        size_t external_head_stride = 0, const int32_t* external_row_offsets = nullptr,
                                        size_t external_row_offsets_head_stride = 0) {
  return patternSparseAttentionRangeTyped(
      params, query, key, value, output, 0, static_cast<size_t>(params.batch) * params.query_heads, 0, params.query_len,
      external_indices, external_head_stride, external_row_offsets, external_row_offsets_head_stride);
}

namespace detail {

struct HmxF16SparseScratch {
  std::vector<float> probabilities;
  std::vector<mllm_fp16_t> query;
  std::vector<mllm_fp16_t> weights;
  std::vector<int32_t> row_indices;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
  std::vector<float> paired_probabilities;
  std::vector<int32_t> paired_indices;
#endif

  void resize(size_t key_len, size_t head_dim) {
    probabilities.resize(key_len);
    query.resize(head_dim);
    weights.resize(key_len);
  }
};

#if defined(MLLM_ANDROID_SME_SPARSE_PV)
struct SparsePvTile {
  std::vector<mllm_fp16_t> panel;
  std::vector<uint64_t> active_keys;
  std::vector<uint64_t> second_block_keys;
  std::vector<int32_t> keys;
  std::vector<mllm_fp16_t> tile_weights;
  std::array<size_t, kSparsePvRows + 1> tile_offsets{};
  bool panel_needs_clear = false;

  void reserve(int32_t key_len, bool shared) {
    const auto key_count = static_cast<size_t>(key_len);
    const auto words = key_count * (shared ? kSparsePvSharedRows : kSparsePvRows);
    // Resizing can expose old weights under a different stride even at the same byte size.
    if (!panel.empty() && (keys.size() != key_count || panel.size() != words)) panel_needs_clear = true;
    panel.resize(words);
    active_keys.resize((static_cast<size_t>(key_len) + 63) / 64);
    if (shared) second_block_keys.resize(active_keys.size());
    keys.resize(key_len);
  }

  void start(bool retire) {
    if (!retire || panel_needs_clear) std::fill_n(panel.begin(), keys.size() * kSparsePvRows, mllm_fp16_t(0));
    std::fill(active_keys.begin(), active_keys.end(), uint64_t(0));
    // An unfinished tile must be cleared before reuse.
    panel_needs_clear = true;
  }

  void startBitmap(size_t weights) {
    panel_needs_clear = true;
    tile_weights.resize(weights);
    tile_offsets[0] = 0;
    std::fill(active_keys.begin(), active_keys.end(), uint64_t(0));
  }

  mllm_fp16_t* bitmapWeightRow(int32_t row, int32_t keep) {
    const auto begin = tile_offsets[row];
    const auto end = begin + static_cast<size_t>(keep);
    if (end > tile_weights.size()) return nullptr;
    tile_offsets[row + 1] = end;
    return tile_weights.data() + begin;
  }

  template<bool TrackKeys = true>
  void addRow(int32_t row, const int32_t* selected, const mllm_fp16_t* weights, int32_t keep) {
    for (int32_t index = 0; index < keep; ++index) {
      const auto key = static_cast<size_t>(selected[index]);
      panel[key * kSparsePvRows + row] = weights[index];
      if constexpr (TrackKeys) { active_keys[key / 64] |= uint64_t(1) << (key % 64); }
    }
  }

  void startShared() {
    std::fill(panel.begin(), panel.end(), mllm_fp16_t(0));
    std::fill(active_keys.begin(), active_keys.end(), uint64_t(0));
    std::fill(second_block_keys.begin(), second_block_keys.end(), uint64_t(0));
    panel_needs_clear = true;
  }

  std::vector<uint64_t>& sharedMembership(int32_t row) {
    return row < kSparsePvRows ? active_keys : second_block_keys;
  }

  template<bool TrackKeys = true>
  void addSharedRow(int32_t row, const int32_t* selected, const mllm_fp16_t* weights, int32_t keep) {
    for (int32_t index = 0; index < keep; ++index) {
      const auto key = static_cast<size_t>(selected[index]);
      panel[key * kSparsePvSharedRows + row] = weights[index];
      if constexpr (TrackKeys) { sharedMembership(row)[key / 64] |= uint64_t(1) << (key % 64); }
    }
  }

  void computeShared(const mllm_fp16_t* values, float* output) {
    int32_t count = 0;
    for (size_t word = 0; word < active_keys.size(); ++word) {
      const auto first = active_keys[word];
      const auto second = second_block_keys[word];
      for (uint64_t bits = first | second; bits != 0; bits &= bits - 1) {
        const auto bit = std::countr_zero(bits);
        uint32_t encoded = static_cast<uint32_t>(word * 64 + bit);
        encoded |= (first & (uint64_t(1) << bit)) ? kSparsePvSharedFirst : 0;
        encoded |= (second & (uint64_t(1) << bit)) ? kSparsePvSharedSecond : 0;
        keys[count++] = static_cast<int32_t>(encoded);
      }
    }
    sparsePvSharedRowsSme(values, keys.data(), panel.data(), count, output);
  }

  int32_t collectActiveKeys() {
    int32_t count = 0;
    for (size_t word = 0; word < active_keys.size(); ++word) {
      uint64_t pending = active_keys[word];
      while (pending != 0) {
        keys[count++] = static_cast<int32_t>(word * 64 + std::countr_zero(pending));
        pending &= pending - 1;
      }
    }
    return count;
  }

  void compute(const mllm_fp16_t* values, float* output, bool retire) {
    const int32_t count = collectActiveKeys();
    if (retire) {
      sparsePvRowsSmeRetirePanel(values, keys.data(), panel.data(), count, output, useSparsePvOrderedKeys());
      panel_needs_clear = false;
    } else {
      sparsePvRowsSme(values, keys.data(), panel.data(), count, output, useSparsePvOrderedKeys());
    }
  }

  bool computeBitmap(const HmxTopKBitmapRows& bitmap, int32_t first_row, const mllm_fp16_t* values, float* output) {
    const int32_t count = collectActiveKeys();
    for (int32_t i = 0; i < count; ++i) {
      std::fill_n(panel.data() + static_cast<size_t>(keys[i]) * kSparsePvRows, kSparsePvRows, mllm_fp16_t(0));
    }
    if (!bitmap.scatterWeights<mllm_fp16_t, kSparsePvRows>(first_row, tile_offsets, tile_weights, panel)) return false;
    sparsePvRowsSme(values, keys.data(), panel.data(), count, output, useSparsePvOrderedKeys());
    return true;
  }
};
#endif

template<bool TrackMax>
inline float selectedQKScoresF16(const mllm_fp16_t* query, const mllm_fp16_t* key_base, int32_t key_stride,
                                 const int32_t* selected, int32_t keep, int32_t head_dim, float scale, float* scores) {
  float row_max = -std::numeric_limits<float>::infinity();
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
  int32_t index = 0;
  for (; index + 8 <= keep; index += 8) {
    if (index + 16 <= keep) {
      for (int32_t future = 0; future < 8; ++future) {
        const mllm_fp16_t* future_key = key_base + static_cast<size_t>(selected[index + 8 + future]) * key_stride;
        for (int32_t dim = 0; dim < head_dim; dim += 32) { __builtin_prefetch(future_key + dim, 0, 1); }
      }
    }
    const mllm_fp16_t* keys[8] = {
        key_base + static_cast<size_t>(selected[index]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 1]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 2]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 3]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 4]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 5]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 6]) * key_stride,
        key_base + static_cast<size_t>(selected[index + 7]) * key_stride,
    };
    float32x4_t acc_low[8] = {vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                              vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F)};
    float32x4_t acc_high[8] = {vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                               vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F)};
    int32_t dim = 0;
    for (; dim + 8 <= head_dim; dim += 8) {
      const float16x8_t q = vld1q_f16(reinterpret_cast<const float16_t*>(query + dim));
      for (int32_t lane = 0; lane < 8; ++lane) {
        const float16x8_t k = vld1q_f16(reinterpret_cast<const float16_t*>(keys[lane] + dim));
        acc_low[lane] = vfmlalq_low_f16(acc_low[lane], q, k);
        acc_high[lane] = vfmlalq_high_f16(acc_high[lane], q, k);
      }
    }
    for (int32_t lane = 0; lane < 8; ++lane) {
      float sum = vaddvq_f32(vaddq_f32(acc_low[lane], acc_high[lane]));
      for (int32_t tail = dim; tail < head_dim; ++tail) {
        sum += static_cast<float>(query[tail]) * static_cast<float>(keys[lane][tail]);
      }
      const float score = sum * scale;
      scores[index + lane] = score;
      if constexpr (TrackMax) { row_max = std::max(row_max, score); }
    }
  }
  for (; index < keep; ++index) {
    const mllm_fp16_t* key = key_base + static_cast<size_t>(selected[index]) * key_stride;
    float32x4_t acc_low = vdupq_n_f32(0.0F);
    float32x4_t acc_high = vdupq_n_f32(0.0F);
    int32_t dim = 0;
    for (; dim + 8 <= head_dim; dim += 8) {
      const float16x8_t q = vld1q_f16(reinterpret_cast<const float16_t*>(query + dim));
      const float16x8_t k = vld1q_f16(reinterpret_cast<const float16_t*>(key + dim));
      acc_low = vfmlalq_low_f16(acc_low, q, k);
      acc_high = vfmlalq_high_f16(acc_high, q, k);
    }
    float sum = vaddvq_f32(vaddq_f32(acc_low, acc_high));
    for (; dim < head_dim; ++dim) { sum += static_cast<float>(query[dim]) * static_cast<float>(key[dim]); }
    const float score = sum * scale;
    scores[index] = score;
    if constexpr (TrackMax) { row_max = std::max(row_max, score); }
  }
#else
  for (int32_t index = 0; index < keep; ++index) {
    ggml::vec_dot_fp16(head_dim, scores + index, query, key_base + static_cast<size_t>(selected[index]) * key_stride);
    scores[index] *= scale;
    if constexpr (TrackMax) { row_max = std::max(row_max, scores[index]); }
  }
#endif
  return row_max;
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
inline constexpr int kPairQkWidth = 128;

inline bool bitmapPairQkOption(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

template<bool First, bool Second>
inline void bitmapPairDot128(const mllm_fp16_t* a, const mllm_fp16_t* b, const mllm_fp16_t* key,
                            float scale, float* score_a, float* score_b) {
  float32x4_t a_low{}, a_high{}, b_low{}, b_high{};
  for (int dim = 0; dim < kPairQkWidth; dim += 8) {
    const auto k = vld1q_f16(reinterpret_cast<const float16_t*>(key + dim));
    if constexpr (First) {
      const auto q = vld1q_f16(reinterpret_cast<const float16_t*>(a + dim));
      a_low = vfmlalq_low_f16(a_low, q, k);
      a_high = vfmlalq_high_f16(a_high, q, k);
    }
    if constexpr (Second) {
      const auto q = vld1q_f16(reinterpret_cast<const float16_t*>(b + dim));
      b_low = vfmlalq_low_f16(b_low, q, k);
      b_high = vfmlalq_high_f16(b_high, q, k);
    }
  }
  if constexpr (First) *score_a = vaddvq_f32(vaddq_f32(a_low, a_high)) * scale;
  if constexpr (Second) *score_b = vaddvq_f32(vaddq_f32(b_low, b_high)) * scale;
}

template<bool Trace>
inline bool scoreBitmapPair(const HmxTopKBitmapRows& bitmap, int32_t row, int32_t keep, int32_t next_keep,
                            const float* query, int32_t query_stride, const mllm_fp16_t* key, float scale,
                            std::span<uint64_t> active_keys, HmxF16SparseScratch& scratch, size_t& shared) {
  std::array<mllm_fp16_t, kPairQkWidth> next_query;
  mllm_fp32_to_fp16_row(query, scratch.query.data(), kPairQkWidth);
  mllm_fp32_to_fp16_row(query + query_stride, next_query.data(), kPairQkWidth);
  return bitmap.visitPair(row, keep, next_keep, active_keys, [&](int32_t token, bool a, bool b, size_t i, size_t j) {
    const auto* key_row = key + static_cast<size_t>(token) * kPairQkWidth;
    if (a) {
      scratch.row_indices[i] = token;
      if (b) {
        scratch.paired_indices[j] = token;
        bitmapPairDot128<true, true>(scratch.query.data(), next_query.data(), key_row, scale,
                                    scratch.probabilities.data() + i, scratch.paired_probabilities.data() + j);
        if constexpr (Trace) ++shared;
      } else {
        bitmapPairDot128<true, false>(scratch.query.data(), next_query.data(), key_row, scale,
                                     scratch.probabilities.data() + i, nullptr);
      }
    } else {
      scratch.paired_indices[j] = token;
      bitmapPairDot128<false, true>(scratch.query.data(), next_query.data(), key_row, scale,
                                   nullptr, scratch.paired_probabilities.data() + j);
    }
  });
}
#endif

inline void normalizeSoftmax(float* probabilities, int32_t keep) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
  arm::softmax_v1_fp32(probabilities, probabilities, keep, 1, 1);
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
  x86::softmax_v1_fp32(probabilities, probabilities, keep, 1, 1);
#else
  float maximum = -std::numeric_limits<float>::infinity();
  for (int32_t index = 0; index < keep; ++index) { maximum = std::max(maximum, probabilities[index]); }
  float sum = 0.0F;
  for (int32_t index = 0; index < keep; ++index) {
    probabilities[index] = std::exp(probabilities[index] - maximum);
    sum += probabilities[index];
  }
  const float inverse_sum = 1.0F / sum;
  for (int32_t index = 0; index < keep; ++index) { probabilities[index] *= inverse_sum; }
#endif
}

inline void sparseValueF16(const mllm_fp16_t* packed_value, const int32_t* selected, const mllm_fp16_t* weights, int32_t keep,
                           int32_t head_dim, float* output) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
  constexpr int32_t kDimTile = 64;
  int32_t dim = 0;
  for (; dim + kDimTile <= head_dim; dim += kDimTile) {
    float32x4_t accumulators[16] = {
        vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
        vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
        vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
    };
    for (int32_t index = 0; index < keep; ++index) {
      if (index + 8 < keep) {
        __builtin_prefetch(packed_value + static_cast<size_t>(selected[index + 8]) * head_dim + dim, 0, 1);
      }
      const mllm_fp16_t* value_row = packed_value + static_cast<size_t>(selected[index]) * head_dim + dim;
      const float16x8_t weight = vdupq_n_f16(weights[index]);
      for (int32_t block = 0; block < 8; ++block) {
        const float16x8_t value = vld1q_f16(reinterpret_cast<const float16_t*>(value_row + block * 8));
        accumulators[block * 2] = vfmlalq_low_f16(accumulators[block * 2], value, weight);
        accumulators[block * 2 + 1] = vfmlalq_high_f16(accumulators[block * 2 + 1], value, weight);
      }
    }
    for (int32_t block = 0; block < 16; ++block) { vst1q_f32(output + dim + block * 4, accumulators[block]); }
  }
  for (; dim + 8 <= head_dim; dim += 8) {
    float32x4_t acc_low = vdupq_n_f32(0.0F);
    float32x4_t acc_high = vdupq_n_f32(0.0F);
    for (int32_t index = 0; index < keep; ++index) {
      const mllm_fp16_t* value_row = packed_value + static_cast<size_t>(selected[index]) * head_dim + dim;
      const float16x8_t value = vld1q_f16(reinterpret_cast<const float16_t*>(value_row));
      const float16x8_t weight = vdupq_n_f16(weights[index]);
      acc_low = vfmlalq_low_f16(acc_low, value, weight);
      acc_high = vfmlalq_high_f16(acc_high, value, weight);
    }
    vst1q_f32(output + dim, acc_low);
    vst1q_f32(output + dim + 4, acc_high);
  }
  for (; dim < head_dim; ++dim) {
    float sum = 0.0F;
    for (int32_t index = 0; index < keep; ++index) {
      sum += static_cast<float>(weights[index])
             * static_cast<float>(packed_value[static_cast<size_t>(selected[index]) * head_dim + dim]);
    }
    output[dim] = sum;
  }
#else
  for (int32_t dim = 0; dim < head_dim; ++dim) {
    float sum = 0.0F;
    for (int32_t index = 0; index < keep; ++index) {
      sum += static_cast<float>(weights[index])
             * static_cast<float>(packed_value[static_cast<size_t>(selected[index]) * head_dim + dim]);
    }
    output[dim] = sum;
  }
#endif
}

}  // namespace detail

inline bool patternSparseAttentionHmxF16Range(const PatternSparseAttentionF32Params& params, const float* query,
                                              const mllm_fp16_t* key, const mllm_fp16_t* value, float* output,
                                              size_t matrix_begin, size_t matrix_count, int32_t query_row_begin,
                                              int32_t query_row_count, const int32_t* external_indices,
                                              size_t external_head_stride, const int32_t* external_row_offsets,
                                              size_t external_row_offsets_head_stride, HmxTopKBitmapView bitmap = {}) {
  if (query == nullptr || key == nullptr || value == nullptr || output == nullptr || external_indices == nullptr
      || external_row_offsets == nullptr || params.batch <= 0 || params.query_heads <= 0 || params.kv_heads <= 0
      || params.query_heads % params.kv_heads != 0 || params.query_len <= 0 || params.key_len < params.query_len
      || params.query_key_dim <= 0 || params.value_dim != params.query_key_dim || params.query_stride[3] != 1
      || params.key_stride[2] != params.query_key_dim || params.key_stride[3] != 1 || params.value_stride[2] != params.value_dim
      || params.value_stride[3] != 1 || params.output_stride[3] != 1 || external_head_stride == 0
      || external_row_offsets_head_stride == 0) {
    return false;
  }
  const size_t total_matrices = static_cast<size_t>(params.batch) * params.query_heads;
  if (matrix_begin > total_matrices || matrix_count > total_matrices - matrix_begin || query_row_begin < 0
      || query_row_begin > params.query_len || query_row_count < 0 || query_row_count > params.query_len - query_row_begin) {
    return false;
  }
  if (matrix_count == 0 || query_row_count == 0) { return true; }

  thread_local detail::HmxF16SparseScratch scratch;
  scratch.resize(static_cast<size_t>(params.key_len), static_cast<size_t>(params.query_key_dim));
  const bool use_bitmap = bitmap.data != nullptr;
  HmxTopKBitmapRows bitmap_rows;
  if (use_bitmap) {
    if (matrix_count != 1 || !params.causal_mask || params.query_len <= 1
        || !bitmap_rows.reset(bitmap, params.query_len, params.key_len)) {
      return false;
    }
    scratch.row_indices.resize(params.key_len);
  }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
  static const bool pair_option = detail::bitmapPairQkOption("MLLM_HMX_BITMAP_PAIR_QK");
  static const bool pair_trace = detail::bitmapPairQkOption("MLLM_HMX_BITMAP_PAIR_QK_TRACE");
  // Pair only inside this callback and inside the same 16- or 32-row PV tile.
  const bool pair_qk = pair_option && use_bitmap && params.query_key_dim == detail::kPairQkWidth
                       && query_row_begin % 2 == 0 && query_row_count >= 2;
  if (pair_qk) {
    scratch.paired_indices.resize(params.key_len);
    scratch.paired_probabilities.resize(params.key_len);
  }
  size_t paired_rows = 0, paired_positions = 0, shared_keys = 0;
#endif
#if defined(MLLM_ANDROID_SME_SPARSE_PV)
  // HMX publishes unique per-row indices. The opt-in SME reduction uses the
  // same FP16 probabilities for finite values, grouped sixteen rows at a time.
  const bool use_sme = params.value_dim == kSparsePvChannels && params.output_stride[2] == kSparsePvChannels
                       && query_row_count >= kSparsePvRows && useSparsePvSme();
  const bool use_bitmap_panel = use_sme && use_bitmap && useSparsePvBitmapPanel();
  const bool retire_panel = use_sme && (!use_bitmap || use_bitmap_panel) && useSparsePvRetiringPanel();
  const int32_t sme_end = query_row_begin + (use_sme ? query_row_count / kSparsePvRows * kSparsePvRows : 0);
  const bool compatible_panel = use_bitmap ? use_bitmap_panel && useSparsePvBitmapSharedValues() : !retire_panel;
  const bool share_values = use_sme && compatible_panel && useSparsePvOrderedKeys()
                            && useSparsePvSharedValues() && query_row_count >= kSparsePvSharedRows
                            && static_cast<uint32_t>(params.key_len - 1) <= kSparsePvSharedKeyMask;
  const int32_t shared_end = query_row_begin + (share_values ? query_row_count / kSparsePvSharedRows * kSparsePvSharedRows : 0);
  thread_local detail::SparsePvTile pv_tile;
  if (use_sme) pv_tile.reserve(params.key_len, share_values);
#endif
  const int32_t groups = params.query_heads / params.kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(params.query_key_dim));
  for (size_t matrix = matrix_begin; matrix < matrix_begin + matrix_count; ++matrix) {
    const int32_t batch = static_cast<int32_t>(matrix / static_cast<size_t>(params.query_heads));
    const int32_t query_head = static_cast<int32_t>(matrix % static_cast<size_t>(params.query_heads));
    const int32_t kv_head = query_head / groups;
    const float density = params.head_retentions == nullptr ? 1.0F - params.sparsity : params.head_retentions[query_head];
    if (density < 0.0F || density > 1.0F) { return false; }
    const mllm_fp16_t* key_base =
        key + static_cast<size_t>(batch) * params.key_stride[0] + static_cast<size_t>(kv_head) * params.key_stride[1];
    const mllm_fp16_t* value_base =
        value + static_cast<size_t>(batch) * params.value_stride[0] + static_cast<size_t>(kv_head) * params.value_stride[1];
    const int32_t* row_offsets = external_row_offsets + static_cast<size_t>(query_head) * external_row_offsets_head_stride;
    const int32_t query_row_end = query_row_begin + query_row_count;
    const auto visibleKeys = [&](int32_t row) {
      return params.causal_mask && params.query_len > 1
          ? std::min(params.key_len, params.key_len - params.query_len + row + 1) : params.key_len;
    };
    const auto rowKeep = [&](int32_t row, int32_t valid) {
      const auto expected = valid <= params.dense_tokens
          ? valid : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
      const int32_t begin = row_offsets[row], end = row_offsets[row + 1];
      if (begin < 0 || end <= begin || end - begin != expected || static_cast<size_t>(end) > external_head_stride) return 0;
      return end - begin;
    };
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
    bool paired_row_ready = false;
#endif
    for (int32_t query_index = query_row_begin; query_index < query_row_end; ++query_index) {
      const int32_t valid = visibleKeys(query_index);
      const int32_t keep = rowKeep(query_index, valid);
      if (keep == 0) return false;
      const int32_t begin = row_offsets[query_index];
      std::span<uint64_t> active_keys;
      mllm_fp16_t* weights = scratch.weights.data();
#if defined(MLLM_ANDROID_SME_SPARSE_PV)
      const bool shared_row = query_index < shared_end;
      const bool bitmap_panel_row = use_bitmap_panel && !shared_row && query_index < sme_end;
      const bool deferred_bitmap_row = bitmap_panel_row && !retire_panel;
      const int32_t lane = (query_index - query_row_begin) % kSparsePvRows;
      const int32_t shared_lane = (query_index - query_row_begin) % kSparsePvSharedRows;
      if (shared_row) {
        if (shared_lane == 0) pv_tile.startShared();
        // Decode merges the producer words into the membership for this 16-row half.
        if (use_bitmap) active_keys = pv_tile.sharedMembership(shared_lane);
      } else if (bitmap_panel_row) {
        active_keys = pv_tile.active_keys;
        if (deferred_bitmap_row) {
          if (lane == 0) {
            const int32_t tile_end = row_offsets[query_index + kSparsePvRows];
            if (tile_end < row_offsets[query_index + 1] || static_cast<size_t>(tile_end) > external_head_stride) return false;
            pv_tile.startBitmap(static_cast<size_t>(tile_end - begin));
          }
          weights = pv_tile.bitmapWeightRow(lane, keep);
          if (weights == nullptr) return false;
        } else if (lane == 0) {
          pv_tile.start(true);
        }
      }
#endif
      const float* query_row = query + static_cast<size_t>(batch) * params.query_stride[0]
                               + static_cast<size_t>(query_head) * params.query_stride[1]
                               + static_cast<size_t>(query_index) * params.query_stride[2];
      const int32_t* selected;
      float* probabilities = scratch.probabilities.data();
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
      if (paired_row_ready) {
        selected = scratch.paired_indices.data();
        probabilities = scratch.paired_probabilities.data();
        paired_row_ready = false;
      } else if (pair_qk && query_index % 2 == 0 && query_index + 1 < query_row_end) {
        const int32_t next_keep = rowKeep(query_index + 1, visibleKeys(query_index + 1));
        if (next_keep == 0) return false;
        const auto score_pair = pair_trace ? detail::scoreBitmapPair<true> : detail::scoreBitmapPair<false>;
        if (!score_pair(bitmap_rows, query_index, keep, next_keep, query_row, params.query_stride[2],
                        key_base, scale, active_keys, scratch, shared_keys)) return false;
        if (pair_trace) {
          paired_rows += 2;
          paired_positions += static_cast<size_t>(keep) + next_keep;
        }
        selected = scratch.row_indices.data();
        paired_row_ready = true;
      } else
#endif
      {
        if (use_bitmap) {
          if (!bitmap_rows.decode(query_index, {scratch.row_indices.data(), static_cast<size_t>(keep)}, active_keys)) return false;
          selected = scratch.row_indices.data();
        } else {
          selected = external_indices + matrix * external_head_stride + begin;
          for (int32_t index = 0; index < keep; ++index) {
            if (selected[index] < 0 || selected[index] >= valid) { return false; }
          }
        }
        mllm_fp32_to_fp16_row(query_row, scratch.query.data(), params.query_key_dim);
        detail::selectedQKScoresF16<true>(scratch.query.data(), key_base, params.key_stride[2], selected, keep,
                                         params.query_key_dim, scale, probabilities);
      }
      detail::normalizeSoftmax(probabilities, keep);
      mllm_fp32_to_fp16_row(probabilities, weights, keep);
      float* output_row = output + static_cast<size_t>(batch) * params.output_stride[0]
                          + static_cast<size_t>(query_head) * params.output_stride[1]
                          + static_cast<size_t>(query_index) * params.output_stride[2];
#if defined(MLLM_ANDROID_SME_SPARSE_PV)
      if (shared_row) {
        if (use_bitmap) pv_tile.addSharedRow<false>(shared_lane, selected, weights, keep);
        else pv_tile.addSharedRow(shared_lane, selected, weights, keep);
        if (shared_lane == kSparsePvSharedRows - 1) {
          pv_tile.computeShared(value_base, output_row - shared_lane * kSparsePvChannels);
        }
        continue;
      }
      if (query_index < sme_end) {
        if (deferred_bitmap_row) {
          if (lane == kSparsePvRows - 1
              && !pv_tile.computeBitmap(bitmap_rows, query_index - lane, value_base, output_row - lane * kSparsePvChannels)) {
            return false;
          }
        } else {
          if (bitmap_panel_row) {
            // Decode already accumulated the active-key mask for this tile.
            pv_tile.addRow<false>(lane, selected, weights, keep);
          } else {
            if (lane == 0) pv_tile.start(retire_panel);
            pv_tile.addRow(lane, selected, weights, keep);
          }
          if (lane == kSparsePvRows - 1) pv_tile.compute(value_base, output_row - lane * kSparsePvChannels, retire_panel);
        }
        continue;
      }
#endif
      detail::sparseValueF16(value_base, selected, weights, keep, params.value_dim, output_row);
    }
  }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
  if (pair_qk && pair_trace) {
    std::fprintf(stderr, "[SHADOWNPU_BITMAP_PAIR_QK] key_len=%d matrix_begin=%zu row_begin=%d pair_rows=%zu positions=%zu shared_keys=%zu\n",
                 params.key_len, matrix_begin, query_row_begin, paired_rows, paired_positions, shared_keys);
  }
#endif
#if defined(MLLM_ANDROID_SME_SPARSE_PV)
  if (use_sme && traceSparsePvSme()) {
    std::fprintf(stderr, "[SHADOWNPU_SME_PV] key_len=%d matrix_begin=%zu matrices=%zu row_begin=%d tiled_rows=%d tail_rows=%d\n",
                 params.key_len, matrix_begin, matrix_count, query_row_begin, sme_end - query_row_begin,
                 query_row_begin + query_row_count - sme_end);
    if (useSparsePvOrderedKeys()) {
      std::fprintf(stderr, "[SHADOWNPU_SME_ORDERED_KEYS] key_len=%d matrix_begin=%zu matrices=%zu row_begin=%d tiled_rows=%d\n",
                   params.key_len, matrix_begin, matrix_count, query_row_begin, sme_end - query_row_begin);
    }
    if (share_values) {
      std::fprintf(stderr, "[SHADOWNPU_SME_SHARED_VALUES] key_len=%d matrix_begin=%zu matrices=%zu row_begin=%d shared_rows=%d\n",
                   params.key_len, matrix_begin, matrix_count, query_row_begin, shared_end - query_row_begin);
    }
    const int32_t remaining_sme_rows = sme_end - shared_end;
    if (use_bitmap_panel && remaining_sme_rows > 0) {
      std::fprintf(stderr, "[SHADOWNPU_SME_BITMAP_PANEL] key_len=%d matrix_begin=%zu row_begin=%d tiled_rows=%d\n",
                   params.key_len, matrix_begin, shared_end, remaining_sme_rows);
    }
    if (retire_panel && remaining_sme_rows > 0) {
      std::fprintf(stderr, "[SHADOWNPU_SME_RETIRE_PANEL] key_len=%d matrix_begin=%zu matrices=%zu row_begin=%d tiled_rows=%d\n",
                   params.key_len, matrix_begin, matrix_count, shared_end, remaining_sme_rows);
      if (use_bitmap_panel) {
        std::fprintf(stderr, "[SHADOWNPU_SME_DIRECT_BITMAP_WEIGHTS] key_len=%d matrix_begin=%zu row_begin=%d tiled_rows=%d\n",
                     params.key_len, matrix_begin, shared_end, remaining_sme_rows);
      }
    }
  }
#endif
  return true;
}

inline bool patternSparseAttentionF32(const PatternSparseAttentionF32Params& params, const float* query, const float* key,
                                      const float* value, float* output) {
  return patternSparseAttentionTyped(params, query, key, value, output);
}

}  // namespace mllm::cpu::shadownpu
