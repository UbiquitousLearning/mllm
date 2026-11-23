// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace mllm::cpu::shadownpu {

struct SparseSoftmaxValueF32Params {
  int32_t batch = 0;
  int32_t query_heads = 0;
  int32_t kv_heads = 0;
  int32_t query_len = 0;
  int32_t key_len = 0;
  int32_t value_dim = 0;
  std::array<int32_t, 4> logits_stride{};
  std::array<int32_t, 4> value_stride{};
  std::array<int32_t, 4> output_stride{};
  float sparsity = 0.0F;
  bool causal_mask = true;
  int32_t dense_tokens = 0;
  const float* head_retentions = nullptr;
};

template<typename ValueType>
inline bool sparseSoftmaxValueTyped(const SparseSoftmaxValueF32Params& params, const float* logits, const ValueType* value,
                                    float* output) {
  if (logits == nullptr || value == nullptr || output == nullptr || params.batch <= 0 || params.query_heads <= 0
      || params.kv_heads <= 0 || params.query_heads % params.kv_heads != 0 || params.query_len <= 0
      || params.key_len < params.query_len || params.value_dim <= 0 || params.sparsity < 0.0F || params.sparsity >= 1.0F
      || params.dense_tokens < 0) {
    return false;
  }

  const int32_t groups = params.query_heads / params.kv_heads;
  std::vector<int32_t> selected(static_cast<size_t>(params.key_len));
  std::vector<float> weights(static_cast<size_t>(params.key_len));

  for (int32_t batch = 0; batch < params.batch; ++batch) {
    for (int32_t query_head = 0; query_head < params.query_heads; ++query_head) {
      const int32_t kv_head = query_head / groups;
      const float density = params.head_retentions == nullptr ? 1.0F - params.sparsity : params.head_retentions[query_head];
      if (!(density > 0.0F && density <= 1.0F)) { return false; }

      for (int32_t query_index = 0; query_index < params.query_len; ++query_index) {
        const int32_t valid = params.causal_mask && params.query_len > 1
                                  ? std::min(params.key_len, params.key_len - params.query_len + query_index + 1)
                                  : params.key_len;
        const int32_t keep =
            valid <= params.dense_tokens
                ? valid
                : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
        const float* logits_row = logits + static_cast<size_t>(batch) * params.logits_stride[0]
                                  + static_cast<size_t>(query_head) * params.logits_stride[1]
                                  + static_cast<size_t>(query_index) * params.logits_stride[2];

        std::iota(selected.begin(), selected.begin() + valid, 0);
        for (int32_t key_index = 0; key_index < valid; ++key_index) {
          if (std::isnan(logits_row[static_cast<size_t>(key_index) * params.logits_stride[3]])) { return false; }
        }
        if (keep < valid) {
          std::partial_sort(selected.begin(), selected.begin() + keep, selected.begin() + valid, [&](int32_t lhs, int32_t rhs) {
            const float lhs_score = logits_row[static_cast<size_t>(lhs) * params.logits_stride[3]];
            const float rhs_score = logits_row[static_cast<size_t>(rhs) * params.logits_stride[3]];
            return lhs_score == rhs_score ? lhs < rhs : lhs_score > rhs_score;
          });
        }

        // Android Release builds use -ffast-math; a finite sentinel keeps
        // softmax valid for the operation's finite-logit execution path.
        float maximum = std::numeric_limits<float>::lowest();
        for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
          maximum = std::max(maximum, logits_row[static_cast<size_t>(selected[selected_index]) * params.logits_stride[3]]);
        }

        float denominator = 0.0F;
        if (std::isinf(maximum)) {
          for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
            const float score = logits_row[static_cast<size_t>(selected[selected_index]) * params.logits_stride[3]];
            weights[selected_index] = score == maximum ? 1.0F : 0.0F;
            denominator += weights[selected_index];
          }
        } else {
          for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
            const float score = logits_row[static_cast<size_t>(selected[selected_index]) * params.logits_stride[3]];
            weights[selected_index] = std::exp(score - maximum);
            denominator += weights[selected_index];
          }
        }
        if (!(denominator > 0.0F)) { return false; }

        float* output_row = output + static_cast<size_t>(batch) * params.output_stride[0]
                            + static_cast<size_t>(query_head) * params.output_stride[1]
                            + static_cast<size_t>(query_index) * params.output_stride[2];
        const float inverse_denominator = 1.0F / denominator;
        for (int32_t dim = 0; dim < params.value_dim; ++dim) {
          float result = 0.0F;
          for (int32_t selected_index = 0; selected_index < keep; ++selected_index) {
            const int32_t key_index = selected[selected_index];
            const ValueType* value_row = value + static_cast<size_t>(batch) * params.value_stride[0]
                                         + static_cast<size_t>(kv_head) * params.value_stride[1]
                                         + static_cast<size_t>(key_index) * params.value_stride[2];
            result += weights[selected_index] * inverse_denominator
                      * static_cast<float>(value_row[static_cast<size_t>(dim) * params.value_stride[3]]);
          }
          output_row[static_cast<size_t>(dim) * params.output_stride[3]] = result;
        }
      }
    }
  }
  return true;
}

}  // namespace mllm::cpu::shadownpu
