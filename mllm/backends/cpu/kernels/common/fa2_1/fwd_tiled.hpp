// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "mllm/core/Parallel.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64)
#include <arm_neon.h>
#endif

namespace mllm::cpu::flash_attn2 {

// Canonical indexing B,H,S,D, independent of the tensor's physical layout.
template<typename T>
struct StridedAttentionView {
  T* data;
  std::array<int64_t, 4> strides;

  T* row(int32_t b, int32_t h, int32_t s) const { return data + b * strides[0] + h * strides[1] + s * strides[2]; }
};

struct TiledAttentionShape {
  int32_t batch, query_heads, kv_heads, query_tokens, kv_tokens, query_dim, value_dim;
  bool causal;
  int32_t sliding_window;  // 0 means no window; positive windows require causal attention.
};

namespace details {

inline constexpr int32_t kScoreTile = 32;
inline constexpr int32_t kQueryRowsPerJob = 16;

inline void tiledScores(const float* query, int64_t q_stride, const float* keys, int64_t k_stride, int64_t key_step,
                        int32_t dim, int32_t count, float scale, float* scores) {
  int32_t i = 0;
#if defined(MLLM_HOST_ARCH_ARM64)
  if (q_stride == 1 && k_stride == 1) {
    for (; i <= count - 4; i += 4) {
      const float* k0 = keys + i * key_step;
      const float* k1 = k0 + key_step;
      const float* k2 = k1 + key_step;
      const float* k3 = k2 + key_step;
      float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
      int32_t d = 0;
      for (; d <= dim - 4; d += 4) {
        const auto q = vld1q_f32(query + d);
        a0 = vfmaq_f32(a0, q, vld1q_f32(k0 + d));
        a1 = vfmaq_f32(a1, q, vld1q_f32(k1 + d));
        a2 = vfmaq_f32(a2, q, vld1q_f32(k2 + d));
        a3 = vfmaq_f32(a3, q, vld1q_f32(k3 + d));
      }
      float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1), s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
      for (; d < dim; ++d) {
        s0 += query[d] * k0[d];
        s1 += query[d] * k1[d];
        s2 += query[d] * k2[d];
        s3 += query[d] * k3[d];
      }
      scores[i] = s0 * scale;
      scores[i + 1] = s1 * scale;
      scores[i + 2] = s2 * scale;
      scores[i + 3] = s3 * scale;
    }
  }
#endif
  for (; i < count; ++i) {
    const auto* key = keys + i * key_step;
    float sum = 0.f;
    for (int32_t d = 0; d < dim; ++d) sum += query[d * q_stride] * key[d * k_stride];
    scores[i] = sum * scale;
  }
}

inline void tiledValues(float* output, int64_t o_stride, const float* values, int64_t v_stride, int64_t value_step, int32_t dim,
                        const float* weights, int32_t count, float previous_scale) {
  int32_t d = 0;
#if defined(MLLM_HOST_ARCH_ARM64)
  if (o_stride == 1 && v_stride == 1) {
    for (; d <= dim - 16; d += 16) {
      auto a0 = vmulq_n_f32(vld1q_f32(output + d), previous_scale);
      auto a1 = vmulq_n_f32(vld1q_f32(output + d + 4), previous_scale);
      auto a2 = vmulq_n_f32(vld1q_f32(output + d + 8), previous_scale);
      auto a3 = vmulq_n_f32(vld1q_f32(output + d + 12), previous_scale);
      for (int32_t i = 0; i < count; ++i) {
        const auto* row = values + i * value_step + d;
        const auto p = weights[i];
        a0 = vfmaq_n_f32(a0, vld1q_f32(row), p);
        a1 = vfmaq_n_f32(a1, vld1q_f32(row + 4), p);
        a2 = vfmaq_n_f32(a2, vld1q_f32(row + 8), p);
        a3 = vfmaq_n_f32(a3, vld1q_f32(row + 12), p);
      }
      vst1q_f32(output + d, a0);
      vst1q_f32(output + d + 4, a1);
      vst1q_f32(output + d + 8, a2);
      vst1q_f32(output + d + 12, a3);
    }
  }
#endif
  for (; d < dim; ++d) {
    float sum = output[d * o_stride] * previous_scale;
    for (int32_t i = 0; i < count; ++i) sum += weights[i] * values[i * value_step + d * v_stride];
    output[d * o_stride] = sum;
  }
}

inline void tiledQueryRow(const TiledAttentionShape& shape, StridedAttentionView<const float> query,
                          StridedAttentionView<const float> key, StridedAttentionView<const float> value,
                          StridedAttentionView<float> output, int32_t b, int32_t h, int32_t key_head, int32_t s, float scale) {
  const int32_t end = shape.causal ? shape.kv_tokens - shape.query_tokens + s + 1 : shape.kv_tokens;
  const int32_t begin = shape.sliding_window > 0 ? std::max(0, end - shape.sliding_window) : 0;
  const float* q = query.row(b, h, s);
  float* out = output.row(b, h, s);
  for (int32_t d = 0; d < shape.value_dim; ++d) out[d * output.strides[3]] = 0.f;
  float maximum = -std::numeric_limits<float>::infinity(), denominator = 0.f;
  std::array<float, kScoreTile> probabilities;
  for (int32_t start = begin; start < end;) {
    const int32_t count = std::min(end - start, kScoreTile);
    tiledScores(q, query.strides[3], key.row(b, key_head, start), key.strides[3], key.strides[2], shape.query_dim, count, scale,
                probabilities.data());
    const float next_max = std::max(maximum, *std::max_element(probabilities.begin(), probabilities.begin() + count));
    const float previous_scale = maximum == next_max ? 1.f : std::exp(maximum - next_max);
    denominator *= previous_scale;
    maximum = next_max;
    for (int32_t i = 0; i < count; ++i) {
      probabilities[i] = std::exp(probabilities[i] - maximum);
      denominator += probabilities[i];
    }
    tiledValues(out, output.strides[3], value.row(b, key_head, start), value.strides[3], value.strides[2], shape.value_dim,
                probabilities.data(), count, previous_scale);
    start += count;
  }
  const float inverse = 1.f / denominator;
  for (int32_t d = 0; d < shape.value_dim; ++d) out[d * output.strides[3]] *= inverse;
}

}  // namespace details

// Callers validate shape compatibility, causal suffix bounds, and disjoint
// output. Scratch is bounded by one score tile per worker, not sequence length.
inline void fwdTiledFp32(const TiledAttentionShape& shape, StridedAttentionView<const float> query,
                         StridedAttentionView<const float> key, StridedAttentionView<const float> value,
                         StridedAttentionView<float> output, int32_t threads) {
  const int32_t rows_per_job = details::kQueryRowsPerJob;
  const int64_t row_jobs = 1 + (shape.query_tokens - 1) / rows_per_job;
  const int64_t head_jobs = static_cast<int64_t>(shape.batch) * shape.query_heads;
  const int32_t group_size = shape.query_heads / shape.kv_heads;
  const float scale = 1.f / std::sqrt(static_cast<float>(shape.query_dim));
  MLLM_CONDITIONAL_PARALLEL_FOR(threads > 1, threads, job, 0, row_jobs * head_jobs, 1, {
    const int64_t head_job = job % head_jobs;
    const auto b = static_cast<int32_t>(head_job / shape.query_heads);
    const auto h = static_cast<int32_t>(head_job % shape.query_heads);
    const auto first = static_cast<int32_t>(job / head_jobs * rows_per_job);
    const auto last = first + std::min(shape.query_tokens - first, rows_per_job);
    for (int32_t s = first; s < last; ++s) {
      details::tiledQueryRow(shape, query, key, value, output, b, h, h / group_size, s, scale);
    }
  });
}

}  // namespace mllm::cpu::flash_attn2
