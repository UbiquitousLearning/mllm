// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/common/kda/kimi_delta_attention.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/core/aops/KimiDeltaAttentionOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

void referenceKDA(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<float>& gate, const std::vector<float>& beta, const std::vector<float>& a_log,
                  const std::vector<float>& dt_bias, std::vector<float>& state, std::vector<float>& output, int batch,
                  int sequence, int heads, int dim, bool safe_gate, float lower_bound) {
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(dim));
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const std::size_t state_base = (static_cast<std::size_t>(b) * heads + h) * dim * dim;
      for (int s = 0; s < sequence; ++s) {
        const std::size_t vector_base = ((static_cast<std::size_t>(b) * sequence + s) * heads + h) * dim;
        float q_norm_sq = 0.0F;
        float k_norm_sq = 0.0F;
        for (int d = 0; d < dim; ++d) {
          q_norm_sq += q[vector_base + d] * q[vector_base + d];
          k_norm_sq += k[vector_base + d] * k[vector_base + d];
        }
        const float q_scale = query_dim_scale / std::sqrt(q_norm_sq + 1.0e-6F);
        const float k_scale = 1.0F / std::sqrt(k_norm_sq + 1.0e-6F);
        std::vector<float> normalized_q(dim);
        std::vector<float> normalized_k(dim);
        std::vector<float> prediction(dim, 0.0F);
        for (int d = 0; d < dim; ++d) {
          normalized_q[d] = q[vector_base + d] * q_scale;
          normalized_k[d] = k[vector_base + d] * k_scale;
        }
        for (int kd = 0; kd < dim; ++kd) {
          const float gate_input = gate[vector_base + kd] + dt_bias[h * dim + kd];
          const float log_decay = safe_gate ? lower_bound * sigmoid(std::exp(a_log[h]) * gate_input)
                                            : -std::exp(a_log[h]) * std::log1p(std::exp(gate_input));
          const float decay = std::exp(log_decay);
          for (int vd = 0; vd < dim; ++vd) {
            const std::size_t state_index = state_base + kd * dim + vd;
            state[state_index] *= decay;
            prediction[vd] += normalized_k[kd] * state[state_index];
          }
        }
        for (int vd = 0; vd < dim; ++vd) {
          const float delta =
              beta[(static_cast<std::size_t>(b) * sequence + s) * heads + h] * (v[vector_base + vd] - prediction[vd]);
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

std::vector<float> makeValues(std::size_t count, float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = offset + scale * static_cast<float>((static_cast<int>(i * 37U % 29U) - 14));
  }
  return values;
}

void ensureCpuContext() {
  static const bool kContextInitialized = [] {
    mllm::initializeContext();
    mllm::Context::instance().setCpuOpThreads(1);
    return true;
  }();
  (void)kContextInitialized;
}

TEST(Ling3KDA, MatchesSafeGateReference) {
  constexpr int kBatch = 2;
  constexpr int kSequence = 5;
  constexpr int kHeads = 3;
  constexpr int kDim = 8;
  const std::size_t vector_count = static_cast<std::size_t>(kBatch) * kSequence * kHeads * kDim;
  const std::size_t state_count = static_cast<std::size_t>(kBatch) * kHeads * kDim * kDim;

  auto q = makeValues(vector_count, 0.017F);
  auto k = makeValues(vector_count, -0.013F, 0.02F);
  auto v = makeValues(vector_count, 0.011F, -0.03F);
  auto gate = makeValues(vector_count, 0.019F, 0.1F);
  auto beta = makeValues(static_cast<std::size_t>(kBatch) * kSequence * kHeads, 0.01F, 0.45F);
  auto a_log = makeValues(kHeads, 0.02F, 0.3F);
  auto dt_bias = makeValues(static_cast<std::size_t>(kHeads) * kDim, 0.015F, -0.1F);
  auto expected_state = makeValues(state_count, 0.003F);
  auto actual_state = expected_state;
  std::vector<float> expected_output(vector_count, 0.0F);
  std::vector<float> actual_output(vector_count, 0.0F);

  referenceKDA(q, k, v, gate, beta, a_log, dt_bias, expected_state, expected_output, kBatch, kSequence, kHeads, kDim, true,
               -5.0F);
  mllm::cpu::kda::kimiDeltaAttentionF32(q.data(), k.data(), v.data(), gate.data(), beta.data(), a_log.data(), dt_bias.data(),
                                        actual_state.data(), actual_output.data(), kBatch, kSequence, kHeads, kDim, true, -5.0F,
                                        1);

  for (std::size_t i = 0; i < vector_count; ++i) { EXPECT_NEAR(actual_output[i], expected_output[i], 2.0e-6F); }
  for (std::size_t i = 0; i < state_count; ++i) { EXPECT_NEAR(actual_state[i], expected_state[i], 2.0e-6F); }
}

TEST(Ling3KDA, PrefillAndTokenwiseDecodeAreEquivalent) {
  constexpr int kBatch = 1;
  constexpr int kSequence = 7;
  constexpr int kHeads = 2;
  constexpr int kDim = 8;
  const std::size_t vector_count = static_cast<std::size_t>(kBatch) * kSequence * kHeads * kDim;
  const std::size_t state_count = static_cast<std::size_t>(kBatch) * kHeads * kDim * kDim;
  auto q = makeValues(vector_count, 0.012F);
  auto k = makeValues(vector_count, 0.009F, -0.02F);
  auto v = makeValues(vector_count, -0.008F, 0.01F);
  auto gate = makeValues(vector_count, 0.014F);
  auto beta = makeValues(static_cast<std::size_t>(kBatch) * kSequence * kHeads, 0.008F, 0.5F);
  auto a_log = makeValues(kHeads, 0.03F, 0.2F);
  auto dt_bias = makeValues(static_cast<std::size_t>(kHeads) * kDim, 0.01F, -0.05F);
  std::vector<float> prefill_state(state_count, 0.0F);
  std::vector<float> decode_state(state_count, 0.0F);
  std::vector<float> prefill_output(vector_count, 0.0F);
  std::vector<float> decode_output(vector_count, 0.0F);

  mllm::cpu::kda::kimiDeltaAttentionF32(q.data(), k.data(), v.data(), gate.data(), beta.data(), a_log.data(), dt_bias.data(),
                                        prefill_state.data(), prefill_output.data(), kBatch, kSequence, kHeads, kDim, true,
                                        -5.0F, 1);
  const std::size_t token_width = static_cast<std::size_t>(kHeads) * kDim;
  for (int token = 0; token < kSequence; ++token) {
    const std::size_t offset = static_cast<std::size_t>(token) * token_width;
    mllm::cpu::kda::kimiDeltaAttentionF32(q.data() + offset, k.data() + offset, v.data() + offset, gate.data() + offset,
                                          beta.data() + static_cast<std::size_t>(token) * kHeads, a_log.data(), dt_bias.data(),
                                          decode_state.data(), decode_output.data() + offset, kBatch, 1, kHeads, kDim, true,
                                          -5.0F, 1);
  }

  EXPECT_EQ(prefill_output, decode_output);
  EXPECT_EQ(prefill_state, decode_state);
}

TEST(Ling3KDA, RejectsInvalidSafeGateBound) {
  std::vector<float> values(4, 0.0F);
  std::vector<float> state(4, 0.0F);
  EXPECT_THROW(mllm::cpu::kda::kimiDeltaAttentionF32(values.data(), values.data(), values.data(), values.data(), values.data(),
                                                     values.data(), values.data(), state.data(), values.data(), 1, 1, 1, 2,
                                                     true, 0.0F, 1),
               std::invalid_argument);
}

TEST(Ling3KDA, FunctionalDispatchReturnsOutputAndUpdatedState) {
  ensureCpuContext();
  constexpr int kBatch = 1;
  constexpr int kSequence = 3;
  constexpr int kHeads = 2;
  constexpr int kDim = 4;
  const std::size_t vector_count = static_cast<std::size_t>(kBatch) * kSequence * kHeads * kDim;
  const std::size_t state_count = static_cast<std::size_t>(kBatch) * kHeads * kDim * kDim;

  const auto q = makeValues(vector_count, 0.017F);
  const auto k = makeValues(vector_count, -0.013F, 0.02F);
  const auto v = makeValues(vector_count, 0.011F, -0.03F);
  const auto gate = makeValues(vector_count, 0.019F, 0.1F);
  const auto beta = makeValues(static_cast<std::size_t>(kBatch) * kSequence * kHeads, 0.01F, 0.45F);
  const auto a_log = makeValues(kHeads, 0.02F, 0.3F);
  const auto dt_bias = makeValues(static_cast<std::size_t>(kHeads) * kDim, 0.015F, -0.1F);
  const auto initial_state = makeValues(state_count, 0.003F);
  auto expected_state = initial_state;
  std::vector<float> expected_output(vector_count, 0.0F);
  referenceKDA(q, k, v, gate, beta, a_log, dt_bias, expected_state, expected_output, kBatch, kSequence, kHeads, kDim, true,
               -5.0F);
  auto state_input = mllm::Tensor::fromVector(initial_state, {kBatch, kHeads, kDim, kDim});

  auto [output, updated_state] = mllm::nn::functional::kimiDeltaAttention(
      mllm::Tensor::fromVector(q, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(k, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(v, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(gate, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(beta, {kBatch, kSequence, kHeads}), mllm::Tensor::fromVector(a_log, {kHeads}),
      mllm::Tensor::fromVector(dt_bias, {kHeads * kDim}), state_input, true, -5.0F);

  const auto actual_output = output.toVector<float>();
  const auto actual_state = updated_state.toVector<float>();
  EXPECT_EQ(state_input.toVector<float>(), initial_state);
  for (std::size_t i = 0; i < vector_count; ++i) { EXPECT_NEAR(actual_output[i], expected_output[i], 2.0e-6F); }
  for (std::size_t i = 0; i < state_count; ++i) { EXPECT_NEAR(actual_state[i], expected_state[i], 2.0e-6F); }
}

TEST(Ling3KDA, FunctionalDispatchMatchesKernelAtTinyProductionGeometry) {
  ensureCpuContext();
  constexpr int kBatch = 1;
  constexpr int kSequence = 49;
  constexpr int kHeads = 16;
  constexpr int kDim = 128;
  const std::size_t vector_count = static_cast<std::size_t>(kBatch) * kSequence * kHeads * kDim;
  const std::size_t state_count = static_cast<std::size_t>(kBatch) * kHeads * kDim * kDim;
  const auto q = makeValues(vector_count, 0.0017F);
  const auto k = makeValues(vector_count, -0.0013F, 0.002F);
  const auto v = makeValues(vector_count, 0.0011F, -0.003F);
  const auto gate = makeValues(vector_count, 0.0019F, 0.01F);
  const auto beta = makeValues(static_cast<std::size_t>(kBatch) * kSequence * kHeads, 0.001F, 0.45F);
  const auto a_log = makeValues(kHeads, 0.002F, 0.3F);
  const auto dt_bias = makeValues(static_cast<std::size_t>(kHeads) * kDim, 0.0015F, -0.1F);
  const auto initial_state = makeValues(state_count, 0.0003F);
  auto expected_state = initial_state;
  std::vector<float> expected_output(vector_count, 0.0F);
  mllm::cpu::kda::kimiDeltaAttentionF32(q.data(), k.data(), v.data(), gate.data(), beta.data(), a_log.data(), dt_bias.data(),
                                        expected_state.data(), expected_output.data(), kBatch, kSequence, kHeads, kDim, true,
                                        -5.0F, 8);

  auto [output, updated_state] = mllm::nn::functional::kimiDeltaAttention(
      mllm::Tensor::fromVector(q, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(k, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(v, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(gate, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(beta, {kBatch, kSequence, kHeads}), mllm::Tensor::fromVector(a_log, {kHeads}),
      mllm::Tensor::fromVector(dt_bias, {kHeads * kDim}), mllm::Tensor::fromVector(initial_state, {kBatch, kHeads, kDim, kDim}),
      true, -5.0F);

  EXPECT_EQ(output.toVector<float>(), expected_output);
  EXPECT_EQ(updated_state.toVector<float>(), expected_state);
}

TEST(Ling3KDA, FunctionalInplaceStateAliasesAndMutatesInput) {
  ensureCpuContext();
  constexpr int kBatch = 1;
  constexpr int kSequence = 3;
  constexpr int kHeads = 2;
  constexpr int kDim = 4;
  const std::size_t vector_count = static_cast<std::size_t>(kBatch) * kSequence * kHeads * kDim;
  const std::size_t state_count = static_cast<std::size_t>(kBatch) * kHeads * kDim * kDim;
  const auto q = makeValues(vector_count, 0.017F);
  const auto k = makeValues(vector_count, -0.013F, 0.02F);
  const auto v = makeValues(vector_count, 0.011F, -0.03F);
  const auto gate = makeValues(vector_count, 0.019F, 0.1F);
  const auto beta = makeValues(static_cast<std::size_t>(kBatch) * kSequence * kHeads, 0.01F, 0.45F);
  const auto a_log = makeValues(kHeads, 0.02F, 0.3F);
  const auto dt_bias = makeValues(static_cast<std::size_t>(kHeads) * kDim, 0.015F, -0.1F);
  const auto initial_state = makeValues(state_count, 0.003F);
  auto expected_state = initial_state;
  std::vector<float> expected_output(vector_count, 0.0F);
  mllm::cpu::kda::kimiDeltaAttentionF32(q.data(), k.data(), v.data(), gate.data(), beta.data(), a_log.data(), dt_bias.data(),
                                        expected_state.data(), expected_output.data(), kBatch, kSequence, kHeads, kDim, true,
                                        -5.0F, 8);
  auto state = mllm::Tensor::fromVector(initial_state, {kBatch, kHeads, kDim, kDim});

  auto [output, updated_state] = mllm::nn::functional::kimiDeltaAttention(
      mllm::Tensor::fromVector(q, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(k, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(v, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(gate, {kBatch, kSequence, kHeads, kDim}),
      mllm::Tensor::fromVector(beta, {kBatch, kSequence, kHeads}), mllm::Tensor::fromVector(a_log, {kHeads}),
      mllm::Tensor::fromVector(dt_bias, {kHeads * kDim}), state, true, -5.0F, true);

  EXPECT_EQ(updated_state.ptr<float>(), state.ptr<float>());
  EXPECT_EQ(output.toVector<float>(), expected_output);
  EXPECT_EQ(state.toVector<float>(), expected_state);
}

TEST(Ling3KDA, FunctionalDispatchProducesLinalgIR) {
  ensureCpuContext();
  auto q = mllm::Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU);
  auto beta = mllm::Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU);
  auto a_log = mllm::Tensor::empty({2}, mllm::kFloat32, mllm::kCPU);
  auto dt_bias = mllm::Tensor::empty({8}, mllm::kFloat32, mllm::kCPU);
  auto state = mllm::Tensor::empty({1, 2, 4, 4}, mllm::kFloat32, mllm::kCPU);

  mllm::ir::lowlevel::traceStart();
  auto outputs = mllm::nn::functional::kimiDeltaAttention(q, q, q, q, beta, a_log, dt_bias, state, true, -5.0F, true);
  auto ir_context = mllm::ir::lowlevel::traceStop();

  EXPECT_EQ(outputs[0].shape(), q.shape());
  EXPECT_EQ(outputs[1].shape(), state.shape());
  mllm::ir::linalg::LinalgIROp::ptr_t kda_ir;
  for (const auto& op : ir_context->topLevelOp()->cast_<mllm::ir::Op>()->getTopRegion()->ops()) {
    if (op->isa_<mllm::ir::linalg::KimiDeltaAttentionOp>()) {
      kda_ir = op->cast_<mllm::ir::linalg::LinalgIROp>();
      break;
    }
  }
  ASSERT_NE(kda_ir, nullptr);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(kda_ir);
  EXPECT_EQ(options.at("safe_gate"), true);
  EXPECT_FLOAT_EQ(options.at("lower_bound").get<float>(), -5.0F);
  EXPECT_EQ(options.at("state_inplace"), true);
  const auto restored =
      mllm::jit::interpreter::aopsFromJson(nlohmann::json{{"op_type", "KimiDeltaAttention"}, {"op_options", options}});
  const auto restored_kda = std::dynamic_pointer_cast<mllm::aops::KimiDeltaAttentionOp>(restored);
  ASSERT_NE(restored_kda, nullptr);
  EXPECT_EQ(restored_kda->options().safe_gate, true);
  EXPECT_FLOAT_EQ(restored_kda->options().lower_bound, -5.0F);
  EXPECT_EQ(restored_kda->options().state_inplace, true);
}

}  // namespace
