// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"

namespace {

using mllm::Tensor;

class KimiDeltaAttentionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class KimiDeltaAttentionModule final : public mllm::nn::Module {
 public:
  KimiDeltaAttentionModule(std::string name, bool safe_gate, float lower_bound, bool state_inplace) : Module(std::move(name)) {
    kda_ = reg<mllm::nn::KimiDeltaAttention>("kda", safe_gate, lower_bound, state_inplace);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    auto [output, state] = kda_(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]);
    return {output, state};
  }

 private:
  mllm::nn::KimiDeltaAttention kda_;
};

template<typename OpType>
auto findOp(const mllm::ir::node_ptr_t& node) -> typename OpType::ptr_t {
  if (node->isa_<OpType>()) { return node->cast_<OpType>(); }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findOp<OpType>(op)) { return found; }
    }
  }
  return nullptr;
}

Tensor patterned(const Tensor::shape_t& shape, float scale, float offset = 0.0F) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < tensor.numel(); ++index) {
    tensor.ptr<float>()[index] = std::sin(static_cast<float>(index + 1) * scale) + offset;
  }
  return tensor;
}

float stableSigmoid(float value) {
  if (value >= 0.0F) {
    const float exp_value = std::exp(-value);
    return 1.0F / (1.0F + exp_value);
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

// Independent scalar reference of the public contract:
// q/k/v/gate [B, S, H, D], beta [B, S, H] (post-sigmoid), a_log [H],
// dt_bias [H * D], state [B, H, D, D] -> output [B, S, H, D], new state.
std::pair<std::vector<float>, std::vector<float>> referenceKimiDeltaAttention(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& gate,
    const std::vector<float>& beta, const std::vector<float>& a_log, const std::vector<float>& dt_bias,
    std::vector<float> state, int batch, int sequence, int heads, int dim, bool safe_gate, float lower_bound) {
  const float query_dim_scale = 1.0F / std::sqrt(static_cast<float>(dim));
  std::vector<float> output(static_cast<std::size_t>(batch) * sequence * heads * dim, 0.0F);
  std::vector<float> normalized_q(dim);
  std::vector<float> normalized_k(dim);
  std::vector<float> prediction(dim);
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const std::size_t state_base = (static_cast<std::size_t>(b) * heads + h) * dim * dim;
      const float a_scale = std::exp(a_log[h]);
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
        for (int d = 0; d < dim; ++d) {
          normalized_q[d] = q[vector_base + d] * q_scale;
          normalized_k[d] = k[vector_base + d] * k_scale;
          prediction[d] = 0.0F;
        }
        for (int kd = 0; kd < dim; ++kd) {
          const float gate_input = gate[vector_base + kd] + dt_bias[h * dim + kd];
          const float log_decay = safe_gate ? lower_bound * stableSigmoid(a_scale * gate_input)
                                            : -a_scale * std::log1p(std::exp(gate_input));
          const float decay = std::exp(log_decay);
          for (int vd = 0; vd < dim; ++vd) {
            const std::size_t state_index = state_base + kd * dim + vd;
            state[state_index] *= decay;
            prediction[vd] += normalized_k[kd] * state[state_index];
          }
        }
        const float beta_value = beta[(static_cast<std::size_t>(b) * sequence + s) * heads + h];
        for (int vd = 0; vd < dim; ++vd) {
          const float delta = beta_value * (v[vector_base + vd] - prediction[vd]);
          for (int kd = 0; kd < dim; ++kd) {
            const std::size_t state_index = state_base + kd * dim + vd;
            state[state_index] += normalized_k[kd] * delta;
            output[vector_base + vd] += normalized_q[kd] * state[state_index];
          }
        }
      }
    }
  }
  return {output, state};
}

void expectNear(const Tensor& actual, const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(actual.numel(), expected.size());
  for (int index = 0; index < actual.numel(); ++index) {
    ASSERT_NEAR(actual.ptr<float>()[index], expected[index], tolerance) << "index " << index;
  }
}

struct Case {
  int batch;
  int sequence;
  int heads;
  int dim;
  bool safe_gate;
  float lower_bound;
  float tolerance;
};

void runEagerReferenceCase(const Case& test_case, const std::string& module_name) {
  SCOPED_TRACE(::testing::Message() << module_name << " B=" << test_case.batch << " S=" << test_case.sequence
                                    << " H=" << test_case.heads << " D=" << test_case.dim
                                    << " safe_gate=" << test_case.safe_gate);
  const Tensor::shape_t vector_shape = {test_case.batch, test_case.sequence, test_case.heads, test_case.dim};
  auto q = patterned(vector_shape, 0.03F);
  auto k = patterned(vector_shape, 0.05F, 0.1F);
  auto v = patterned(vector_shape, 0.07F, -0.2F);
  auto gate = patterned(vector_shape, 0.09F, -0.2F);
  auto beta = patterned({test_case.batch, test_case.sequence, test_case.heads}, 0.11F, 0.5F);
  auto a_log = patterned({test_case.heads}, 0.13F, -0.4F);
  auto dt_bias = patterned({test_case.heads * test_case.dim}, 0.15F, -0.1F);
  auto state = patterned({test_case.batch, test_case.heads, test_case.dim, test_case.dim}, 0.017F);
  const auto state_before = state.toVector<float>();
  const auto [expected_output, expected_state] = referenceKimiDeltaAttention(
      q.toVector<float>(), k.toVector<float>(), v.toVector<float>(), gate.toVector<float>(), beta.toVector<float>(),
      a_log.toVector<float>(), dt_bias.toVector<float>(), state_before, test_case.batch, test_case.sequence, test_case.heads,
      test_case.dim, test_case.safe_gate, test_case.lower_bound);

  KimiDeltaAttentionModule module(module_name, test_case.safe_gate, test_case.lower_bound, /*state_inplace=*/false);
  const auto outputs = module(q, k, v, gate, beta, a_log, dt_bias, state);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0].shape(), vector_shape);
  EXPECT_EQ(outputs[1].shape(), state.shape());
  EXPECT_NE(outputs[1].ptr<float>(), state.ptr<float>());
  EXPECT_EQ(state.toVector<float>(), state_before);
  expectNear(outputs[0], expected_output, test_case.tolerance);
  expectNear(outputs[1], expected_state, test_case.tolerance);
}

TEST_F(KimiDeltaAttentionTest, EagerMatchesIndependentReferenceAndPreservesInputState) {
  runEagerReferenceCase({2, 5, 3, 8, true, -5.0F, 2.0e-6F}, "kimi_delta_attention_safe_gate");
  runEagerReferenceCase({1, 4, 2, 8, false, 0.0F, 2.0e-6F}, "kimi_delta_attention_softplus_gate");
  runEagerReferenceCase({1, 3, 2, 6, true, -2.5F, 2.0e-6F}, "kimi_delta_attention_custom_bound");
}

TEST_F(KimiDeltaAttentionTest, EagerMatchesIndependentReferenceAtProductionHeadGeometry) {
  // Ling-3.0-tiny geometry: 16 heads of 128 lanes; the batched lane helpers
  // and multi-threaded (batch, head) partition are all exercised here.
  runEagerReferenceCase({1, 49, 16, 128, true, -5.0F, 5.0e-5F}, "kimi_delta_attention_production");
}

TEST_F(KimiDeltaAttentionTest, InplaceStateOutputAliasesAndMutatesInput) {
  auto q = patterned({1, 3, 2, 4}, 0.03F);
  auto k = patterned(q.shape(), 0.05F);
  auto v = patterned(q.shape(), 0.07F);
  auto gate = patterned(q.shape(), 0.09F, -0.2F);
  auto beta = patterned({1, 3, 2}, 0.11F, 0.5F);
  auto a_log = patterned({2}, 0.13F, -0.4F);
  auto dt_bias = patterned({8}, 0.15F, -0.1F);
  auto state = patterned({1, 2, 4, 4}, 0.017F);
  const auto state_before = state.toVector<float>();
  const auto* state_storage = state.ptr<float>();

  KimiDeltaAttentionModule module("kimi_delta_attention_inplace", true, -5.0F, /*state_inplace=*/true);
  const auto outputs = module(q, k, v, gate, beta, a_log, dt_bias, state);

  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].ptr<float>(), state_storage);
  EXPECT_NE(state.toVector<float>(), state_before);
  EXPECT_EQ(outputs[1].toVector<float>(), state.toVector<float>());
}

TEST_F(KimiDeltaAttentionTest, ChunkedPrefillAndDecodeMatchOneShot) {
  constexpr int kSequence = 7;
  constexpr int kHeads = 2;
  constexpr int kDim = 8;
  auto q = patterned({1, kSequence, kHeads, kDim}, 0.03F);
  auto k = patterned(q.shape(), 0.05F, 0.1F);
  auto v = patterned(q.shape(), 0.07F, -0.2F);
  auto gate = patterned(q.shape(), 0.09F, -0.2F);
  auto beta = patterned({1, kSequence, kHeads}, 0.11F, 0.5F);
  auto a_log = patterned({kHeads}, 0.13F, -0.4F);
  auto dt_bias = patterned({kHeads * kDim}, 0.15F, -0.1F);
  const auto initial_state = patterned({1, kHeads, kDim, kDim}, 0.017F).toVector<float>();

  KimiDeltaAttentionModule one_shot("kimi_delta_attention_one_shot", true, -5.0F, false);
  const auto expected =
      one_shot(q, k, v, gate, beta, a_log, dt_bias, Tensor::fromVector(initial_state, {1, kHeads, kDim, kDim}));
  const auto expected_output = expected[0].toVector<float>();

  KimiDeltaAttentionModule chunked("kimi_delta_attention_chunked", true, -5.0F, false);
  auto state = Tensor::fromVector(initial_state, {1, kHeads, kDim, kDim});
  std::vector<float> chunked_output;
  const std::vector<std::pair<int, int>> chunks = {{0, 3}, {3, 4}, {4, 5}, {5, 7}};
  for (const auto& [begin, end] : chunks) {
    const auto length = end - begin;
    // Token slices [begin, end) of a [1, S, H, D] activation or a [1, S, H] gate.
    const auto slice = [&](const Tensor& tensor, int width, const Tensor::shape_t& shape) {
      const auto values = tensor.toVector<float>();
      const auto first = values.begin() + static_cast<std::ptrdiff_t>(begin) * width;
      return Tensor::fromVector(std::vector<float>(first, first + static_cast<std::ptrdiff_t>(length) * width), shape);
    };
    const Tensor::shape_t vector_shape = {1, length, kHeads, kDim};
    const auto outputs = chunked(slice(q, kHeads * kDim, vector_shape), slice(k, kHeads * kDim, vector_shape),
                                 slice(v, kHeads * kDim, vector_shape), slice(gate, kHeads * kDim, vector_shape),
                                 slice(beta, kHeads, {1, length, kHeads}), a_log, dt_bias, state);
    const auto values = outputs[0].toVector<float>();
    chunked_output.insert(chunked_output.end(), values.begin(), values.end());
    state = outputs[1];
  }

  EXPECT_EQ(chunked_output, expected_output);
  EXPECT_EQ(state.toVector<float>(), expected[1].toVector<float>());
}

TEST_F(KimiDeltaAttentionTest, RejectsInvalidGeometryAndOptions) {
  auto q = patterned({1, 2, 2, 4}, 0.03F);
  auto beta = patterned({1, 2, 2}, 0.11F, 0.5F);
  auto a_log = patterned({2}, 0.13F, -0.4F);
  auto dt_bias = patterned({8}, 0.15F, -0.1F);
  auto state = patterned({1, 2, 4, 4}, 0.017F);

  KimiDeltaAttentionModule module("kimi_delta_attention_invalid", true, -5.0F, false);
  // beta must be [B, S, H].
  EXPECT_THROW((void)module(q, q, q, q, patterned({1, 2, 2, 1}, 0.1F), a_log, dt_bias, state), std::invalid_argument);
  // state must be [B, H, D, D].
  EXPECT_THROW((void)module(q, q, q, q, beta, a_log, dt_bias, patterned({1, 2, 4, 3}, 0.1F)), std::invalid_argument);
  // dt_bias must hold H * D values.
  EXPECT_THROW((void)module(q, q, q, q, beta, a_log, patterned({2}, 0.1F), state), std::invalid_argument);
  // The safe gate requires a finite, strictly negative lower bound.
  KimiDeltaAttentionModule invalid_bound("kimi_delta_attention_invalid_bound", true, 0.0F, false);
  EXPECT_THROW((void)invalid_bound(q, q, q, q, beta, a_log, dt_bias, state), std::invalid_argument);
}

TEST_F(KimiDeltaAttentionTest, TraceAndSerializationPreserveGateAndStateOptions) {
  KimiDeltaAttentionModule module("kimi_delta_attention_trace", true, -3.5F, true);
  auto ir_context = mllm::ir::trace(
      module, Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 2, 2, 4}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({1, 2, 2}, mllm::kFloat32, mllm::kCPU), Tensor::empty({2}, mllm::kFloat32, mllm::kCPU),
      Tensor::empty({8}, mllm::kFloat32, mllm::kCPU), Tensor::empty({1, 2, 4, 4}, mllm::kFloat32, mllm::kCPU));
  auto op = findOp<mllm::ir::linalg::KimiDeltaAttentionOp>(ir_context->topLevelOp());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->getAOp()->getOpType(), mllm::OpTypes::kKimiDeltaAttention);
  EXPECT_EQ(op->inputs().size(), 8);
  EXPECT_EQ(op->outputs().size(), 2);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_TRUE(options.at("safe_gate").get<bool>());
  EXPECT_FLOAT_EQ(options.at("lower_bound").get<float>(), -3.5F);
  EXPECT_TRUE(options.at("state_inplace").get<bool>());

  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "KimiDeltaAttention"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kKimiDeltaAttention);
  const auto restored_options = std::static_pointer_cast<mllm::aops::KimiDeltaAttentionOp>(restored)->options();
  EXPECT_TRUE(restored_options.safe_gate);
  EXPECT_FLOAT_EQ(restored_options.lower_bound, -3.5F);
  EXPECT_TRUE(restored_options.state_inplace);
}

}  // namespace
