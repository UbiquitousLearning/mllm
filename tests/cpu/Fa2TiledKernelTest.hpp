// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "mllm/backends/cpu/kernels/common/fa2_1/fwd_tiled.hpp"
#include "tests/cpu/KernelTestHelper.hpp"

namespace fa2_tiled_kernel_test {

using mllm::cpu::flash_attn2::fwdTiledFp32;
using mllm::cpu::flash_attn2::StridedAttentionView;
using mllm::cpu::flash_attn2::TiledAttentionShape;

inline std::array<int64_t, 4> storageStrides(int heads, int seq, int dim, bool sequence_first, int channel_stride) {
  // Pad the feature rows and reserve unused sequence capacity in every batch.
  const int64_t row = (dim + 3LL) * channel_stride;
  const int64_t capacity = seq + 5;
  return sequence_first ? std::array<int64_t, 4>{heads * capacity * row, row, heads * row, channel_stride}
                        : std::array<int64_t, 4>{heads * capacity * row, capacity * row, row, channel_stride};
}

inline void fillPattern(std::vector<float>& values, int salt, float scale) {
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = scale * static_cast<float>(static_cast<int>((i * salt + i / 7) % 53) - 26) / 32.f;
  }
}

inline void testAgainstDoubleReference(TiledAttentionShape shape, bool sequence_first, int channel_stride, int threads,
                                       float logit_scale) {
  const auto qs = storageStrides(shape.query_heads, shape.query_tokens, shape.query_dim, sequence_first, channel_stride);
  const auto ks = storageStrides(shape.kv_heads, shape.kv_tokens, shape.query_dim, sequence_first, channel_stride);
  const auto vs = storageStrides(shape.kv_heads, shape.kv_tokens, shape.value_dim, sequence_first, channel_stride);
  const auto os = storageStrides(shape.query_heads, shape.query_tokens, shape.value_dim, sequence_first, channel_stride);
  std::vector<float> q(shape.batch * qs[0]), k(shape.batch * ks[0]), v(shape.batch * vs[0]);
  constexpr float kUnwritten = -1234.f;
  std::vector<float> got(shape.batch * os[0], kUnwritten);
  std::vector<bool> touched(got.size());
  fillPattern(q, 13, logit_scale);
  fillPattern(k, 17, logit_scale);
  fillPattern(v, 19, 1.f);
  const StridedAttentionView<const float> query{q.data(), qs}, key{k.data(), ks}, value{v.data(), vs};
  const StridedAttentionView<float> output{got.data(), os};
  fwdTiledFp32(shape, query, key, value, output, threads);

  std::vector<double> logits(shape.kv_tokens);
  const auto scale = 1.0 / std::sqrt(static_cast<double>(shape.query_dim));
  for (int b = 0; b < shape.batch; ++b) {
    for (int h = 0; h < shape.query_heads; ++h) {
      const int kh = h / (shape.query_heads / shape.kv_heads);
      for (int s = 0; s < shape.query_tokens; ++s) {
        const int end = shape.causal ? shape.kv_tokens - shape.query_tokens + s + 1 : shape.kv_tokens;
        const int begin = shape.sliding_window == 0 ? 0 : std::max(0, end - shape.sliding_window);
        double max_score = -std::numeric_limits<double>::infinity();
        for (int i = begin; i < end; ++i) {
          double sum = 0.0;
          for (int d = 0; d < shape.query_dim; ++d) {
            sum += static_cast<double>(query.row(b, h, s)[d * qs[3]]) * key.row(b, kh, i)[d * ks[3]];
          }
          logits[i] = sum * scale;
          max_score = std::max(max_score, logits[i]);
        }
        double denominator = 0.0;
        for (int i = begin; i < end; ++i) {
          logits[i] = std::exp(logits[i] - max_score);
          denominator += logits[i];
        }
        for (int d = 0; d < shape.value_dim; ++d) {
          double sum = 0.0;
          for (int i = begin; i < end; ++i) sum += logits[i] * value.row(b, kh, i)[d * vs[3]];
          const double reference = sum / denominator;
          const auto* actual = output.row(b, h, s) + d * os[3];
          ASSERT_NEAR(*actual, reference, 1e-4 + 1e-4 * std::abs(reference))
              << "batch=" << b << " head=" << h << " query=" << s << " channel=" << d;
          touched[actual - got.data()] = true;
        }
      }
    }
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (!touched[i]) ASSERT_EQ(got[i], kUnwritten);
  }
}

}  // namespace fa2_tiled_kernel_test

class Fa2TiledKernelTest : public KernelTest {};
