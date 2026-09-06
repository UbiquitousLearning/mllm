// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/kda/kimi_delta_attention.hpp"

#include "mllm/engine/Context.hpp"
#include "mllm/core/Parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm::cpu::kda {
namespace {

constexpr int kMaxParallelLanes = 8;
constexpr std::size_t kMinParallelWork = 65536;

int availableCpuCount(int fallback) {
#if defined(__linux__) || defined(__ANDROID__)
  cpu_set_t affinity = {};
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) { return CPU_COUNT(&affinity); }
#endif
  return fallback;
}

float stableSigmoid(float value) {
  if (value >= 0.0F) {
    const float exp_value = std::exp(-value);
    return 1.0F / (1.0F + exp_value);
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

float stableSoftplus(float value) {
  if (value > 20.0F) { return value; }
  if (value < -20.0F) { return std::exp(value); }
  return std::log1p(std::exp(value));
}

float squaredNorm(const float* values, int length) {
  float sum = 0.0F;
  int index = 0;
#if defined(__aarch64__)
  float32x4_t vector_sum = vdupq_n_f32(0.0F);
  for (; index + 4 <= length; index += 4) {
    const float32x4_t value = vld1q_f32(values + index);
    vector_sum = vmlaq_f32(vector_sum, value, value);
  }
  sum = vaddvq_f32(vector_sum);
#endif
  for (; index < length; ++index) { sum += values[index] * values[index]; }
  return sum;
}

void normalize(const float* source, float scale, float* destination, int length) {
  int index = 0;
#if defined(__aarch64__)
  for (; index + 4 <= length; index += 4) { vst1q_f32(destination + index, vmulq_n_f32(vld1q_f32(source + index), scale)); }
#endif
  for (; index < length; ++index) { destination[index] = source[index] * scale; }
}

void decayAndAccumulatePrediction(float* state_row, float decay, float key, float* prediction, int length) {
  int index = 0;
#if defined(__aarch64__)
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vmulq_n_f32(vld1q_f32(state_row + index), decay);
    vst1q_f32(state_row + index, state_value);
    float32x4_t predicted = vld1q_f32(prediction + index);
    predicted = vmlaq_n_f32(predicted, state_value, key);
    vst1q_f32(prediction + index, predicted);
  }
#endif
  for (; index < length; ++index) {
    state_row[index] *= decay;
    prediction[index] += key * state_row[index];
  }
}

void updateAndAccumulateOutput(float* state_row, float key, float query, const float* delta, float* output, int length) {
  int index = 0;
#if defined(__aarch64__)
  for (; index + 4 <= length; index += 4) {
    float32x4_t state_value = vld1q_f32(state_row + index);
    state_value = vmlaq_n_f32(state_value, vld1q_f32(delta + index), key);
    vst1q_f32(state_row + index, state_value);
    float32x4_t output_value = vld1q_f32(output + index);
    output_value = vmlaq_n_f32(output_value, state_value, query);
    vst1q_f32(output + index, output_value);
  }
#endif
  for (; index < length; ++index) {
    state_row[index] += key * delta[index];
    output[index] += query * state_row[index];
  }
}

}  // namespace

void kimiDeltaAttentionF32(const float* q, const float* k, const float* v, const float* gate_logits, const float* beta,
                           const float* a_log, const float* dt_bias, float* state, float* output, int batch_size,
                           int sequence_length, int num_heads, int head_dim, bool safe_gate, float lower_bound) {
  kimiDeltaAttentionF32(q, k, v, gate_logits, beta, a_log, dt_bias, state, output, batch_size, sequence_length, num_heads,
                        head_dim, safe_gate, lower_bound, ::mllm::Context::instance().getCpuOpThreads());
}

void kimiDeltaAttentionF32(const float* q, const float* k, const float* v, const float* gate_logits, const float* beta,
                           const float* a_log, const float* dt_bias, float* state, float* output, int batch_size,
                           int sequence_length, int num_heads, int head_dim, bool safe_gate, float lower_bound,
                           int thread_count) {
  if (q == nullptr || k == nullptr || v == nullptr || gate_logits == nullptr || beta == nullptr || a_log == nullptr
      || dt_bias == nullptr || state == nullptr || output == nullptr) {
    throw std::invalid_argument("Kimi Delta Attention received a null pointer");
  }
  if (batch_size <= 0 || sequence_length <= 0 || num_heads <= 0 || head_dim <= 0 || thread_count <= 0) {
    throw std::invalid_argument("Kimi Delta Attention received an invalid shape");
  }
  if (safe_gate && (!std::isfinite(lower_bound) || lower_bound >= 0.0F)) {
    throw std::invalid_argument("Kimi Delta Attention safe gate requires a finite negative lower bound");
  }

  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const int task_count = batch_size * num_heads;
  const std::size_t work = static_cast<std::size_t>(task_count) * sequence_length * head_dim * head_dim;
  const int parallel_lanes = std::min({thread_count, task_count, availableCpuCount(thread_count), kMaxParallelLanes});
  const bool use_parallel = parallel_lanes > 1 && work >= kMinParallelWork;
  const int scheduled_lanes = use_parallel ? parallel_lanes : 1;

  const auto run_lane = [&](int lane) {
    std::vector<float> normalized_query(static_cast<std::size_t>(head_dim));
    std::vector<float> normalized_key(static_cast<std::size_t>(head_dim));
    std::vector<float> decay(static_cast<std::size_t>(head_dim));
    std::vector<float> prediction(static_cast<std::size_t>(head_dim));
    std::vector<float> delta(static_cast<std::size_t>(head_dim));

    for (int task = lane; task < task_count; task += scheduled_lanes) {
      const int batch = task / num_heads;
      const int head = task % num_heads;
      const std::size_t state_base = (static_cast<std::size_t>(batch) * num_heads + head) * head_dim * head_dim;
      const float a_scale = std::exp(a_log[head]);

      for (int token = 0; token < sequence_length; ++token) {
        const std::size_t vector_base =
            ((static_cast<std::size_t>(batch) * sequence_length + token) * num_heads + head) * head_dim;
        const std::size_t beta_index = (static_cast<std::size_t>(batch) * sequence_length + token) * num_heads + head;

        const float query_scale = query_dim_scale / std::sqrt(squaredNorm(q + vector_base, head_dim) + 1.0e-6F);
        const float key_scale = 1.0F / std::sqrt(squaredNorm(k + vector_base, head_dim) + 1.0e-6F);
        normalize(q + vector_base, query_scale, normalized_query.data(), head_dim);
        normalize(k + vector_base, key_scale, normalized_key.data(), head_dim);

        for (int key_dim = 0; key_dim < head_dim; ++key_dim) {
          const float gate_input = gate_logits[vector_base + key_dim] + dt_bias[head * head_dim + key_dim];
          const float log_decay =
              safe_gate ? lower_bound * stableSigmoid(a_scale * gate_input) : -a_scale * stableSoftplus(gate_input);
          decay[key_dim] = std::exp(log_decay);
        }

        std::fill(prediction.begin(), prediction.end(), 0.0F);
        for (int key_dim = 0; key_dim < head_dim; ++key_dim) {
          float* state_row = state + state_base + static_cast<std::size_t>(key_dim) * head_dim;
          decayAndAccumulatePrediction(state_row, decay[key_dim], normalized_key[key_dim], prediction.data(), head_dim);
        }
        for (int value_dim = 0; value_dim < head_dim; ++value_dim) {
          delta[value_dim] = beta[beta_index] * (v[vector_base + value_dim] - prediction[value_dim]);
        }

        float* token_output = output + vector_base;
        std::fill(token_output, token_output + head_dim, 0.0F);
        for (int key_dim = 0; key_dim < head_dim; ++key_dim) {
          float* state_row = state + state_base + static_cast<std::size_t>(key_dim) * head_dim;
          updateAndAccumulateOutput(state_row, normalized_key[key_dim], normalized_query[key_dim], delta.data(), token_output,
                                    head_dim);
        }
      }
    }
  };

  MLLM_CONDITIONAL_PARALLEL_FOR(use_parallel, scheduled_lanes, lane, 0, scheduled_lanes, 1,
                                { run_lane(static_cast<int>(lane)); });
}

}  // namespace mllm::cpu::kda
