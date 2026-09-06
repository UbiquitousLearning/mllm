// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// Focused oracle for the Kimi Delta Attention (KDA) recurrent kernel.
//
// The reference below is an independent scalar implementation of the frozen
// contract (L2-normalised q/k, 1/sqrt(D) query scaling, safe-gate or softplus
// log-decay, delta-rule state update).  It is deliberately not routed through
// the production kernel, so the NEON row helpers inside kimiDeltaAttentionF32
// cannot validate themselves.  Output and final state are compared separately:
// an output-only comparison would miss a corrupted state that only shows up in
// the next chunk.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mllm/backends/cpu/kernels/common/kda/kimi_delta_attention.hpp"
#include "KernelTestHelper.hpp"

namespace kimi_delta_attention_kernel_test {

using mllm::cpu::kda::kimiDeltaAttentionF32;

struct Geometry {
  int batch = 1;
  int sequence = 1;
  int heads = 1;
  int dim = 1;
  bool safe_gate = true;
  float lower_bound = -5.0F;
  int thread_count = 1;

  [[nodiscard]] std::size_t vectorCount() const {
    return static_cast<std::size_t>(batch) * sequence * heads * dim;
  }
  [[nodiscard]] std::size_t betaCount() const { return static_cast<std::size_t>(batch) * sequence * heads; }
  [[nodiscard]] std::size_t stateCount() const { return static_cast<std::size_t>(batch) * heads * dim * dim; }
};

// Deterministic index-derived fill. No RNG, so every host reproduces the same
// bytes without carrying a seed through the evidence record.
inline float patternValue(std::size_t index, float scale, float offset) {
  return offset + scale * static_cast<float>(static_cast<int>(index * 37U % 29U) - 14);
}

inline std::vector<float> makeBuffer(std::size_t count, float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) { values[index] = patternValue(index, scale, offset); }
  return values;
}

struct Inputs {
  std::vector<float> q, k, v, gate, beta, a_log, dt_bias, state;
};

inline Inputs makeInputs(const Geometry& geometry, float scale = 1.0F) {
  Inputs inputs;
  inputs.q = makeBuffer(geometry.vectorCount(), 0.017F * scale);
  inputs.k = makeBuffer(geometry.vectorCount(), -0.013F * scale, 0.02F * scale);
  inputs.v = makeBuffer(geometry.vectorCount(), 0.011F * scale, -0.03F * scale);
  inputs.gate = makeBuffer(geometry.vectorCount(), 0.019F * scale, 0.1F * scale);
  inputs.beta = makeBuffer(geometry.betaCount(), 0.01F * scale, 0.45F);
  inputs.a_log = makeBuffer(static_cast<std::size_t>(geometry.heads), 0.02F * scale, 0.3F);
  inputs.dt_bias = makeBuffer(static_cast<std::size_t>(geometry.heads) * geometry.dim, 0.015F * scale, -0.1F * scale);
  inputs.state = makeBuffer(geometry.stateCount(), 0.003F * scale);
  return inputs;
}

inline float referenceSigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

// Independent scalar reference for the frozen contract:
// q/k/v/gate [B, S, H, D], beta [B, S, H], a_log [H], dt_bias [H, D],
// state [B, H, D, D] updated in place, output [B, S, H, D].
inline void referenceKimiDeltaAttention(const Inputs& inputs, std::vector<float>& state, std::vector<float>& output,
                                        const Geometry& geometry) {
  const int dim = geometry.dim;
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(dim));
  for (int b = 0; b < geometry.batch; ++b) {
    for (int h = 0; h < geometry.heads; ++h) {
      const std::size_t state_base = (static_cast<std::size_t>(b) * geometry.heads + h) * dim * dim;
      for (int s = 0; s < geometry.sequence; ++s) {
        const std::size_t vector_base = ((static_cast<std::size_t>(b) * geometry.sequence + s) * geometry.heads + h) * dim;
        float q_norm_sq = 0.0F;
        float k_norm_sq = 0.0F;
        for (int d = 0; d < dim; ++d) {
          q_norm_sq += inputs.q[vector_base + d] * inputs.q[vector_base + d];
          k_norm_sq += inputs.k[vector_base + d] * inputs.k[vector_base + d];
        }
        const float q_scale = query_dim_scale / std::sqrt(q_norm_sq + 1.0e-6F);
        const float k_scale = 1.0F / std::sqrt(k_norm_sq + 1.0e-6F);
        std::vector<float> normalized_q(dim);
        std::vector<float> normalized_k(dim);
        std::vector<float> prediction(dim, 0.0F);
        for (int d = 0; d < dim; ++d) {
          normalized_q[d] = inputs.q[vector_base + d] * q_scale;
          normalized_k[d] = inputs.k[vector_base + d] * k_scale;
        }
        for (int kd = 0; kd < dim; ++kd) {
          const float gate_input = inputs.gate[vector_base + kd] + inputs.dt_bias[h * dim + kd];
          const float log_decay = geometry.safe_gate
                                      ? geometry.lower_bound * referenceSigmoid(std::exp(inputs.a_log[h]) * gate_input)
                                      : -std::exp(inputs.a_log[h]) * std::log1p(std::exp(gate_input));
          const float decay = std::exp(log_decay);
          for (int vd = 0; vd < dim; ++vd) {
            const std::size_t state_index = state_base + kd * dim + vd;
            state[state_index] *= decay;
            prediction[vd] += normalized_k[kd] * state[state_index];
          }
        }
        const float beta = inputs.beta[(static_cast<std::size_t>(b) * geometry.sequence + s) * geometry.heads + h];
        for (int vd = 0; vd < dim; ++vd) {
          const float delta = beta * (inputs.v[vector_base + vd] - prediction[vd]);
          for (int kd = 0; kd < dim; ++kd) {
            const std::size_t state_index = state_base + kd * dim + vd;
            state[state_index] += normalized_k[kd] * delta;
            output[vector_base + vd] += normalized_q[kd] * state[state_index];
          }
        }
      }
    }
  }
}

// Output and updated state must both match the scalar reference within a
// small tolerance: the NEON helpers reorder the D-lane accumulations.
inline void testMatchesScalarReference(const std::vector<Geometry>& geometries, float tolerance = 2.0e-6F) {
  for (const auto& geometry : geometries) {
    SCOPED_TRACE(::testing::Message() << "B=" << geometry.batch << " S=" << geometry.sequence << " H=" << geometry.heads
                                      << " D=" << geometry.dim << " safe_gate=" << geometry.safe_gate
                                      << " threads=" << geometry.thread_count);
    const auto inputs = makeInputs(geometry);
    auto expected_state = inputs.state;
    auto actual_state = inputs.state;
    std::vector<float> expected_output(geometry.vectorCount(), 0.0F);
    std::vector<float> actual_output(geometry.vectorCount(), 0.0F);

    referenceKimiDeltaAttention(inputs, expected_state, expected_output, geometry);
    kimiDeltaAttentionF32(inputs.q.data(), inputs.k.data(), inputs.v.data(), inputs.gate.data(), inputs.beta.data(),
                          inputs.a_log.data(), inputs.dt_bias.data(), actual_state.data(), actual_output.data(),
                          geometry.batch, geometry.sequence, geometry.heads, geometry.dim, geometry.safe_gate,
                          geometry.lower_bound, geometry.thread_count);

    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      ASSERT_NEAR(actual_output[i], expected_output[i], tolerance) << "output index " << i;
    }
    for (std::size_t i = 0; i < expected_state.size(); ++i) {
      ASSERT_NEAR(actual_state[i], expected_state[i], tolerance) << "state index " << i;
    }
  }
}

// One-shot prefill and token-by-token decode must be bitwise identical: the
// recurrence has no cross-token reduction, so any drift would be a state bug.
inline void testPrefillAndTokenwiseDecodeAreBitwiseEqual(const Geometry& geometry) {
  ASSERT_EQ(geometry.batch, 1) << "tokenwise replay is defined for batch 1";
  const auto inputs = makeInputs(geometry);
  std::vector<float> prefill_state(geometry.stateCount(), 0.0F);
  std::vector<float> decode_state(geometry.stateCount(), 0.0F);
  std::vector<float> prefill_output(geometry.vectorCount(), 0.0F);
  std::vector<float> decode_output(geometry.vectorCount(), 0.0F);

  kimiDeltaAttentionF32(inputs.q.data(), inputs.k.data(), inputs.v.data(), inputs.gate.data(), inputs.beta.data(),
                        inputs.a_log.data(), inputs.dt_bias.data(), prefill_state.data(), prefill_output.data(),
                        geometry.batch, geometry.sequence, geometry.heads, geometry.dim, geometry.safe_gate,
                        geometry.lower_bound, geometry.thread_count);
  const std::size_t token_width = static_cast<std::size_t>(geometry.heads) * geometry.dim;
  for (int token = 0; token < geometry.sequence; ++token) {
    const std::size_t offset = static_cast<std::size_t>(token) * token_width;
    kimiDeltaAttentionF32(inputs.q.data() + offset, inputs.k.data() + offset, inputs.v.data() + offset,
                          inputs.gate.data() + offset, inputs.beta.data() + static_cast<std::size_t>(token) * geometry.heads,
                          inputs.a_log.data(), inputs.dt_bias.data(), decode_state.data(), decode_output.data() + offset,
                          geometry.batch, 1, geometry.heads, geometry.dim, geometry.safe_gate, geometry.lower_bound,
                          geometry.thread_count);
  }

  EXPECT_EQ(prefill_output, decode_output);
  EXPECT_EQ(prefill_state, decode_state);
}

// Lanes partition (batch, head) tasks and never share state rows, so a
// multi-threaded run must reproduce the serial run bitwise.
inline void testParallelLanesMatchSerialBitwise(const Geometry& geometry) {
  ASSERT_GT(geometry.thread_count, 1);
  const auto inputs = makeInputs(geometry, 0.1F);
  auto serial_state = inputs.state;
  auto parallel_state = inputs.state;
  std::vector<float> serial_output(geometry.vectorCount(), 0.0F);
  std::vector<float> parallel_output(geometry.vectorCount(), 0.0F);

  kimiDeltaAttentionF32(inputs.q.data(), inputs.k.data(), inputs.v.data(), inputs.gate.data(), inputs.beta.data(),
                        inputs.a_log.data(), inputs.dt_bias.data(), serial_state.data(), serial_output.data(), geometry.batch,
                        geometry.sequence, geometry.heads, geometry.dim, geometry.safe_gate, geometry.lower_bound,
                        /*thread_count=*/1);
  kimiDeltaAttentionF32(inputs.q.data(), inputs.k.data(), inputs.v.data(), inputs.gate.data(), inputs.beta.data(),
                        inputs.a_log.data(), inputs.dt_bias.data(), parallel_state.data(), parallel_output.data(),
                        geometry.batch, geometry.sequence, geometry.heads, geometry.dim, geometry.safe_gate,
                        geometry.lower_bound, geometry.thread_count);

  EXPECT_EQ(serial_output, parallel_output);
  EXPECT_EQ(serial_state, parallel_state);
}

inline void testRejectsNullBuffersAndInvalidGeometry() {
  std::vector<float> values(8, 0.0F);
  std::vector<float> state(8, 0.0F);
  const auto call = [&](const float* q, int batch, int sequence, int heads, int dim, bool safe_gate, float lower_bound,
                        int threads) {
    kimiDeltaAttentionF32(q, values.data(), values.data(), values.data(), values.data(), values.data(), values.data(),
                          state.data(), values.data(), batch, sequence, heads, dim, safe_gate, lower_bound, threads);
  };
  EXPECT_THROW(call(nullptr, 1, 1, 1, 2, true, -5.0F, 1), std::invalid_argument);
  EXPECT_THROW(call(values.data(), 0, 1, 1, 2, true, -5.0F, 1), std::invalid_argument);
  EXPECT_THROW(call(values.data(), 1, 1, 1, 2, true, -5.0F, 0), std::invalid_argument);
  // The safe gate needs a finite, strictly negative lower bound.
  EXPECT_THROW(call(values.data(), 1, 1, 1, 2, true, 0.0F, 1), std::invalid_argument);
  EXPECT_THROW(call(values.data(), 1, 1, 1, 2, true, INFINITY, 1), std::invalid_argument);
  // The softplus gate ignores the bound entirely.
  EXPECT_NO_THROW(call(values.data(), 1, 1, 1, 2, false, 0.0F, 1));
}

}  // namespace kimi_delta_attention_kernel_test

class KimiDeltaAttentionKernelTest : public KernelTest {
 public:
  KimiDeltaAttentionKernelTest() = default;
  ~KimiDeltaAttentionKernelTest() override = default;

  void testMatchesScalarReference(const std::vector<kimi_delta_attention_kernel_test::Geometry>& geometries) {
    kimi_delta_attention_kernel_test::testMatchesScalarReference(geometries);
  }

  void testPrefillAndTokenwiseDecodeAreBitwiseEqual(const kimi_delta_attention_kernel_test::Geometry& geometry) {
    kimi_delta_attention_kernel_test::testPrefillAndTokenwiseDecodeAreBitwiseEqual(geometry);
  }

  void testParallelLanesMatchSerialBitwise(const kimi_delta_attention_kernel_test::Geometry& geometry) {
    kimi_delta_attention_kernel_test::testParallelLanesMatchSerialBitwise(geometry);
  }

  void testRejectsNullBuffersAndInvalidGeometry() {
    kimi_delta_attention_kernel_test::testRejectsNullBuffersAndInvalidGeometry();
  }
};
