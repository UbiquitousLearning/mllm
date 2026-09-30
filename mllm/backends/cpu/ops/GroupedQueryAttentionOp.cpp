// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GroupedQueryAttentionOp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gqa_decode/fwd_bhsd.hpp"
#include "mllm/core/Parallel.hpp"

namespace mllm::cpu {
namespace {

template<typename Scalar>
void groupedQueryAttentionDirectStrided(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output,
                                        int32_t sliding_window) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  const auto o_stride = output.stride();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  const int32_t context_offset = k_shape[2] - q_shape[2];
  const int32_t jobs = q_shape[0] * q_shape[1];
  // These tables describe arbitrary sequence strides without rebuilding the
  // address expression in the value reduction. Keep their storage per caller
  // thread so steady-state forwards only resize within the retained capacity.
  static thread_local std::vector<const Scalar*> query_rows;
  static thread_local std::vector<const Scalar*> key_rows;
  static thread_local std::vector<const Scalar*> value_rows;
  static thread_local std::vector<Scalar*> output_rows;
  query_rows.resize(static_cast<size_t>(jobs) * q_shape[2]);
  key_rows.resize(static_cast<size_t>(q_shape[0]) * k_shape[1] * k_shape[2]);
  value_rows.resize(static_cast<size_t>(q_shape[0]) * v_shape[1] * v_shape[2]);
  output_rows.resize(static_cast<size_t>(jobs) * q_shape[2]);
  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t head = 0; head < q_shape[1]; ++head) {
      for (int32_t sequence = 0; sequence < q_shape[2]; ++sequence) {
        const size_t row = (static_cast<size_t>(batch) * q_shape[1] + head) * q_shape[2] + sequence;
        query_rows[row] = query.coffsettedPtr<Scalar>({batch, head, sequence, 0});
        output_rows[row] = output.offsettedPtr<Scalar>({batch, head, sequence, 0});
      }
    }
    for (int32_t head = 0; head < k_shape[1]; ++head) {
      for (int32_t sequence = 0; sequence < k_shape[2]; ++sequence) {
        const size_t row = (static_cast<size_t>(batch) * k_shape[1] + head) * k_shape[2] + sequence;
        key_rows[row] = key.coffsettedPtr<Scalar>({batch, head, sequence, 0});
        value_rows[row] = value.coffsettedPtr<Scalar>({batch, head, sequence, 0});
      }
    }
  }
  const Scalar* const* query_row_data = query_rows.data();
  const Scalar* const* key_row_data = key_rows.data();
  const Scalar* const* value_row_data = value_rows.data();
  Scalar* const* output_row_data = output_rows.data();

  MLLM_AUTO_PARALLEL_FOR_BEGIN(job, 0, jobs, 1) {
    const int32_t batch = job / q_shape[1];
    const int32_t query_head = job % q_shape[1];
    const int32_t kv_head = query_head / groups;
    const size_t query_row_base = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2];
    const size_t key_row_base = (static_cast<size_t>(batch) * k_shape[1] + kv_head) * k_shape[2];
    const size_t value_row_base = (static_cast<size_t>(batch) * v_shape[1] + kv_head) * v_shape[2];
    std::vector<float> scores(static_cast<size_t>(k_shape[2]));
    for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
      const int32_t visible_keys = context_offset + query_index + 1;
      const int32_t first_key = sliding_window > 0 ? std::max(0, visible_keys - sliding_window) : 0;
      const size_t query_row_index = query_row_base + query_index;
      const Scalar* query_row = query_row_data[query_row_index];
      Scalar* output_row = output_row_data[query_row_index];
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t key_index = first_key; key_index < visible_keys; ++key_index) {
        float dot = 0.0F;
        const Scalar* key_row = key_row_data[key_row_base + key_index];
        // Keep the contiguous dot expression separate: folding it into the
        // runtime-stride induction loop changes contraction/codegen, which
        // breaks callers bound to an exact generation-token oracle.
        if (q_stride[3] == 1 && k_stride[3] == 1) {
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            dot += static_cast<float>(query_row[dim]) * static_cast<float>(key_row[dim]);
          }
        } else {
          const Scalar* query_element = query_row;
          const Scalar* key_element = key_row;
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            dot += static_cast<float>(*query_element) * static_cast<float>(*key_element);
            query_element += q_stride[3];
            key_element += k_stride[3];
          }
        }
        scores[key_index] = dot * scale;
        maximum = std::max(maximum, scores[key_index]);
      }

      float denominator = 0.0F;
      for (int32_t key_index = first_key; key_index < visible_keys; ++key_index) {
        scores[key_index] = std::exp(scores[key_index] - maximum);
        denominator += scores[key_index];
      }
      const float inverse_denominator = 1.0F / denominator;
      for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
        float accumulated = 0.0F;
        for (int32_t key_index = first_key; key_index < visible_keys; ++key_index) {
          accumulated +=
              (scores[key_index] * static_cast<float>(value_row_data[value_row_base + key_index][value_dim * v_stride[3]]))
              * inverse_denominator;
        }
        output_row[value_dim * o_stride[3]] = static_cast<Scalar>(accumulated);
      }
    }
  }
  MLLM_AUTO_PARALLEL_FOR_END()
}

// Scalar fallback for the decode variant, used when the vectorized decode
// kernel declines the given geometry.
void groupedQueryAttentionDecodeFloat32Reference(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  // The decode kernel declines any tensor whose last-dimension stride is not 1,
  // so this fallback is reached precisely when that may hold. Honour the output
  // stride rather than assuming a contiguous value dimension.
  const auto o_stride = output.stride();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));

  static thread_local std::vector<float> probabilities;
  probabilities.resize(static_cast<size_t>(k_shape[2]));

  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t query_head = 0; query_head < q_shape[1]; ++query_head) {
      const int32_t kv_head = query_head / groups;
      const auto* q_head = query.coffsettedPtr<float>({batch, query_head, 0, 0});
      const auto* k_head = key.coffsettedPtr<float>({batch, kv_head, 0, 0});
      const auto* v_head = value.coffsettedPtr<float>({batch, kv_head, 0, 0});
      auto* output_head = output.offsettedPtr<float>({batch, query_head, 0, 0});

      // Android release builds use -ffast-math; a finite sentinel keeps stable
      // softmax valid under finite-math assumptions.
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        const auto* key_token = k_head + static_cast<size_t>(key_index) * k_stride[2];
        float score = 0.0F;
        for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
          score += q_head[static_cast<size_t>(dim) * q_stride[3]] * key_token[static_cast<size_t>(dim) * k_stride[3]];
        }
        probabilities[static_cast<size_t>(key_index)] = score * scale;
        maximum = std::max(maximum, probabilities[static_cast<size_t>(key_index)]);
      }

      float denominator = 0.0F;
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        auto& probability = probabilities[static_cast<size_t>(key_index)];
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      const float inverse_denominator = 1.0F / denominator;
      for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
        probabilities[static_cast<size_t>(key_index)] *= inverse_denominator;
      }

      for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
        float result = 0.0F;
        for (int32_t key_index = 0; key_index < k_shape[2]; ++key_index) {
          const auto* value_token = v_head + static_cast<size_t>(key_index) * v_stride[2];
          result += probabilities[static_cast<size_t>(key_index)] * value_token[static_cast<size_t>(value_dim) * v_stride[3]];
        }
        output_head[static_cast<size_t>(value_dim) * o_stride[3]] = result;
      }
    }
  }
}

void groupedQueryAttentionDecodeNativeKV(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const auto q_stride = query.stride();
  const auto k_stride = key.stride();
  const auto v_stride = value.stride();
  const auto output_stride = output.stride();

  static thread_local std::vector<float> probabilities;
  const int32_t group_size = q_shape[1] / k_shape[1];
  probabilities.resize(static_cast<size_t>(group_size) * k_shape[2]);

  const bool completed = cpu::gqa_decode::fwdBhsdFp32(
      q_shape[0], q_shape[1], k_shape[1], k_shape[2], q_shape[3], v_shape[3], query.ptr<float>(),
      {q_stride[0], q_stride[1], q_stride[2], q_stride[3]}, key.ptr<float>(),
      {k_stride[0], k_stride[1], k_stride[2], k_stride[3]}, value.ptr<float>(),
      {v_stride[0], v_stride[1], v_stride[2], v_stride[3]}, output.ptr<float>(),
      {output_stride[0], output_stride[1], output_stride[2], output_stride[3]}, probabilities.data());
  if (!completed) { groupedQueryAttentionDecodeFloat32Reference(query, key, value, output); }
}

}  // namespace

CPUGroupedQueryAttentionOp::CPUGroupedQueryAttentionOp(const aops::GroupedQueryAttentionOpOptions& options)
    : aops::GroupedQueryAttentionOp(options) {}

void CPUGroupedQueryAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (options_.implementation) {
    case aops::GroupedQueryAttentionImplementation::kDirectStrided:
      if (inputs[0].dtype() == kFloat32) {
        groupedQueryAttentionDirectStrided<float>(inputs[0], inputs[1], inputs[2], outputs[0], options_.sliding_window);
      } else {
        groupedQueryAttentionDirectStrided<half_float::half>(inputs[0], inputs[1], inputs[2], outputs[0],
                                                             options_.sliding_window);
      }
      return;
    case aops::GroupedQueryAttentionImplementation::kDecodeNativeKV:
      groupedQueryAttentionDecodeNativeKV(inputs[0], inputs[1], inputs[2], outputs[0]);
      return;
  }
  throw std::invalid_argument("Unsupported CPU GroupedQueryAttention implementation");
}

}  // namespace mllm::cpu
