// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/GroupedQueryAttention.hpp"
#include "mllm/nn/llm_components/GroupedQueryAttention.hpp"

namespace {

using mllm::Tensor;

class GroupedQueryAttentionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class GroupedQueryAttentionDecodeTraceModule final : public mllm::nn::Module {
 public:
  GroupedQueryAttentionDecodeTraceModule() : Module("gqa_decode_trace") {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    return {mllm::nn::llm_components::groupedQueryAttention(inputs[0], inputs[1], inputs[2])};
  }
};

class GroupedQueryAttentionTraceModule final : public mllm::nn::Module {
 public:
  GroupedQueryAttentionTraceModule(mllm::aops::GroupedQueryAttentionImplementation implementation =
                                       mllm::aops::GroupedQueryAttentionImplementation::kDirectStrided,
                                   int window = 512)
      : Module("gqa_trace") {
    op = reg<mllm::nn::GroupedQueryAttention>("attn", implementation, window);
  }
  mllm::nn::GroupedQueryAttention op;

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    return {op(inputs[0], inputs[1], inputs[2])};
  }
};

mllm::ir::linalg::GroupedQueryAttentionOp::ptr_t findGroupedQueryAttentionOp(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::GroupedQueryAttentionOp>()) {
    return node->cast_<mllm::ir::linalg::GroupedQueryAttentionOp>();
  }
  if (!node->isa_<mllm::ir::Op>()) { return nullptr; }
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findGroupedQueryAttentionOp(op)) { return found; }
    }
  }
  return nullptr;
}

Tensor sequential(const Tensor::shape_t& shape, float scale) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < tensor.numel(); ++index) {
    tensor.ptr<float>()[index] = std::sin(static_cast<float>(index + 1) * scale);
  }
  return tensor;
}

Tensor gqaReference(const Tensor& query, const Tensor& key, const Tensor& value, int32_t window = 0) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const int32_t groups = q_shape[1] / k_shape[1];
  const int32_t context_offset = k_shape[2] - q_shape[2];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  auto output = Tensor::zeros({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, mllm::kFloat32, mllm::kCPU);

  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t query_head = 0; query_head < q_shape[1]; ++query_head) {
      const int32_t kv_head = query_head / groups;
      for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
        const int32_t allowed_keys = context_offset + query_index + 1;
        std::vector<float> probabilities(static_cast<size_t>(allowed_keys), -std::numeric_limits<float>::infinity());
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t key_index = (window ? std::max(0, allowed_keys - window) : 0); key_index < allowed_keys; ++key_index) {
          float score = 0.0F;
          for (int32_t dim = 0; dim < q_shape[3]; ++dim) {
            score += *query.cptrAt<float>({batch, query_head, query_index, dim})
                     * *key.cptrAt<float>({batch, kv_head, key_index, dim});
          }
          probabilities[static_cast<size_t>(key_index)] = score * scale;
          maximum = std::max(maximum, probabilities[static_cast<size_t>(key_index)]);
        }
        float denominator = 0.0F;
        for (auto& probability : probabilities) {
          probability = std::exp(probability - maximum);
          denominator += probability;
        }
        for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
          float result = 0.0F;
          for (int32_t key_index = (window ? std::max(0, allowed_keys - window) : 0); key_index < allowed_keys; ++key_index) {
            result += probabilities[static_cast<size_t>(key_index)] / denominator
                      * *value.cptrAt<float>({batch, kv_head, key_index, value_dim});
          }
          *output.ptrAt<float>({batch, query_head, query_index, value_dim}) = result;
        }
      }
    }
  }
  return output;
}

Tensor gqaLegacyDirectStridedReference(const Tensor& query, const Tensor& key, const Tensor& value) {
  const auto q_shape = query.shape();
  const auto k_shape = key.shape();
  const auto v_shape = value.shape();
  const int32_t groups = q_shape[1] / k_shape[1];
  const float scale = 1.0F / std::sqrt(static_cast<float>(q_shape[3]));
  const int32_t context_offset = k_shape[2] - q_shape[2];
  const int32_t jobs = q_shape[0] * q_shape[1];
  auto output = Tensor::zeros({q_shape[0], q_shape[1], q_shape[2], v_shape[3]}, mllm::kFloat32, mllm::kCPU);

  std::vector<const float*> query_rows(static_cast<size_t>(jobs) * q_shape[2]);
  std::vector<const float*> key_rows(static_cast<size_t>(q_shape[0]) * k_shape[1] * k_shape[2]);
  std::vector<const float*> value_rows(static_cast<size_t>(q_shape[0]) * v_shape[1] * v_shape[2]);
  std::vector<float*> output_rows(static_cast<size_t>(jobs) * q_shape[2]);
  for (int32_t batch = 0; batch < q_shape[0]; ++batch) {
    for (int32_t head = 0; head < q_shape[1]; ++head) {
      for (int32_t sequence = 0; sequence < q_shape[2]; ++sequence) {
        const size_t row = (static_cast<size_t>(batch) * q_shape[1] + head) * q_shape[2] + sequence;
        query_rows[row] = query.coffsettedPtr<float>({batch, head, sequence, 0});
        output_rows[row] = output.offsettedPtr<float>({batch, head, sequence, 0});
      }
    }
    for (int32_t head = 0; head < k_shape[1]; ++head) {
      for (int32_t sequence = 0; sequence < k_shape[2]; ++sequence) {
        const size_t row = (static_cast<size_t>(batch) * k_shape[1] + head) * k_shape[2] + sequence;
        key_rows[row] = key.coffsettedPtr<float>({batch, head, sequence, 0});
        value_rows[row] = value.coffsettedPtr<float>({batch, head, sequence, 0});
      }
    }
  }

  for (int32_t job = 0; job < jobs; ++job) {
    const int32_t batch = job / q_shape[1];
    const int32_t query_head = job % q_shape[1];
    const int32_t kv_head = query_head / groups;
    std::vector<float> scores(static_cast<size_t>(k_shape[2]));
    for (int32_t query_index = 0; query_index < q_shape[2]; ++query_index) {
      const int32_t visible_keys = context_offset + query_index + 1;
      float maximum = std::numeric_limits<float>::lowest();
      for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
        float dot = 0.0F;
        const size_t query_row = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index;
        const size_t key_row = (static_cast<size_t>(batch) * k_shape[1] + kv_head) * k_shape[2] + key_index;
        for (int32_t dim = 0; dim < q_shape[3]; ++dim) { dot += query_rows[query_row][dim] * key_rows[key_row][dim]; }
        scores[key_index] = dot * scale;
        maximum = std::max(maximum, scores[key_index]);
      }

      float denominator = 0.0F;
      for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
        scores[key_index] = std::exp(scores[key_index] - maximum);
        denominator += scores[key_index];
      }
      for (int32_t value_dim = 0; value_dim < v_shape[3]; ++value_dim) {
        float accumulated = 0.0F;
        for (int32_t key_index = 0; key_index < visible_keys; ++key_index) {
          const size_t value_row = (static_cast<size_t>(batch) * v_shape[1] + kv_head) * v_shape[2] + key_index;
          accumulated += (scores[key_index] / denominator) * value_rows[value_row][value_dim];
        }
        const size_t output_row = (static_cast<size_t>(batch) * q_shape[1] + query_head) * q_shape[2] + query_index;
        output_rows[output_row][value_dim] = accumulated;
      }
    }
  }
  return output;
}

void expectNear(Tensor actual, Tensor expected, float tolerance = 1e-5F) {
  ASSERT_EQ(actual.shape(), expected.shape());
  const auto actual_cpu = actual.to(mllm::kCPU).contiguous();
  const auto expected_cpu = expected.to(mllm::kCPU).contiguous();
  for (int index = 0; index < actual_cpu.numel(); ++index) {
    EXPECT_NEAR(actual_cpu.ptr<float>()[index], expected_cpu.ptr<float>()[index], tolerance) << "index " << index;
  }
}

TEST_F(GroupedQueryAttentionTest, MatchesRepeatedKVReference) {
  auto query = sequential({1, 4, 3, 5}, 0.07F);
  auto key = sequential({1, 2, 3, 5}, 0.11F);
  auto value = sequential({1, 2, 3, 5}, 0.13F);
  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, RegisteredDirectStridedMatchesReference) {
  auto query = sequential({1, 4, 3, 5}, 0.07F);
  auto key = sequential({1, 2, 3, 5}, 0.11F);
  auto value = sequential({1, 2, 3, 5}, 0.13F);
  const auto actual = mllm::nn::functional::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected, 1e-6F);
}

TEST_F(GroupedQueryAttentionTest, DirectStridedMatchesLegacyReductionAtLfm25Geometry) {
  struct Case {
    int32_t query_length;
    int32_t key_length;
  };

  // Cover both LFM2.5 prefill and decode geometry. Android production builds
  // compile the backend with -ffast-math while this reference remains in the
  // test translation unit, so the portable contract is tight numerical
  // agreement; the full-model device gate separately freezes generated token
  // IDs against the exact incumbent artifact.
  for (const auto test_case : {Case{28, 28}, Case{1, 225}}) {
    auto query = sequential({1, 32, test_case.query_length, 64}, 0.007F);
    auto key = sequential({1, 8, test_case.key_length, 64}, 0.011F);
    auto value = sequential({1, 8, test_case.key_length, 64}, 0.013F);
    const auto actual = mllm::nn::functional::groupedQueryAttention(query, key, value);
    const auto expected = gqaLegacyDirectStridedReference(query, key, value);

    ASSERT_NO_FATAL_FAILURE(expectNear(actual, expected, 1e-6F))
        << "query_length=" << test_case.query_length << " key_length=" << test_case.key_length;
  }
}

TEST_F(GroupedQueryAttentionTest, SlidingOpTraceAndSerializationRoundTrip) {
  GroupedQueryAttentionTraceModule module;
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 4, 2, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 3}, mllm::kFloat32, mllm::kCPU));
  auto ir_op = findGroupedQueryAttentionOp(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);
  EXPECT_EQ(ir_op->getAOp()->getOpType(), mllm::OpTypes::kGroupedQueryAttention);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  EXPECT_EQ(options.at("implementation"), "DirectStrided");
  EXPECT_EQ(options.at("sliding_window"), 512);
  const auto restored = mllm::jit::interpreter::aopsFromJson(
      nlohmann::json{{"op_type", "GroupedQueryAttention"}, {"backend", "CPU"}, {"op_options", options}});
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kGroupedQueryAttention);
}

TEST_F(GroupedQueryAttentionTest, SupportsOneKVHeadAndRejectsIllegalGeometry) {
  auto query = sequential({1, 4, 1, 4}, 0.09F);
  auto key = sequential({1, 1, 2, 4}, 0.12F);
  auto value = sequential({1, 1, 2, 3}, 0.15F);
  const auto output = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  EXPECT_EQ(output.shape(), (Tensor::shape_t{1, 4, 1, 3}));
  expectNear(output, gqaReference(query, key, value));

  auto illegal_query = sequential({1, 3, 1, 4}, 0.09F);
  auto illegal_key = sequential({1, 2, 2, 4}, 0.12F);
  auto illegal_value = sequential({1, 2, 2, 3}, 0.15F);
  EXPECT_THROW(mllm::nn::llm_components::groupedQueryAttention(illegal_query, illegal_key, illegal_value),
               std::invalid_argument);
}

TEST_F(GroupedQueryAttentionTest, DecodeReadsNativeCacheStrideWithoutKVExpansion) {
  auto query = sequential({1, 4, 1, 5}, 0.07F);
  auto key_buffer = sequential({1, 2, 7, 5}, 0.11F);
  auto value_buffer = sequential({1, 2, 7, 3}, 0.13F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 4}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 4}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 4, 1, 3}));
  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, DecodeFallsBackForNonContiguousHeadDimension) {
  auto query_buffer = sequential({1, 4, 1, 10}, 0.07F);
  auto query = query_buffer[{mllm::kAll, mllm::kAll, mllm::kAll, {0, 10, 2}}];
  auto key = sequential({1, 2, 4, 5}, 0.11F);
  auto value = sequential({1, 2, 4, 3}, 0.13F);

  ASSERT_NE(query.stride()[3], 1);
  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  expectNear(actual, expected);
}

TEST_F(GroupedQueryAttentionTest, DecodeStaysFiniteAtMiniCPM5ProductGeometry) {
  auto query = sequential({1, 16, 1, 128}, 0.007F);
  auto key_buffer = sequential({1, 2, 2048, 128}, 0.011F);
  auto value_buffer = sequential({1, 2, 2048, 128}, 0.013F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 16, 1, 128}));
  for (int index = 0; index < actual.numel(); ++index) { EXPECT_TRUE(std::isfinite(actual.ptr<float>()[index])); }
  expectNear(actual, expected, 2e-5F);
}

TEST_F(GroupedQueryAttentionTest, DecodeStaysFiniteAtLfm25ProductGeometry) {
  auto query = sequential({1, 32, 1, 64}, 0.007F);
  auto key_buffer = sequential({1, 8, 2048, 64}, 0.011F);
  auto value_buffer = sequential({1, 8, 2048, 64}, 0.013F);
  auto key = key_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];
  auto value = value_buffer[{mllm::kAll, mllm::kAll, {mllm::kAll, 201}, mllm::kAll}];

  const auto actual = mllm::nn::llm_components::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query, key, value);

  EXPECT_EQ(actual.shape(), (Tensor::shape_t{1, 32, 1, 64}));
  for (int index = 0; index < actual.numel(); ++index) { EXPECT_TRUE(std::isfinite(actual.ptr<float>()[index])); }
  expectNear(actual, expected, 2e-5F);
}

TEST_F(GroupedQueryAttentionTest, DecodeOpTraceAndSerializationRoundTrip) {
  GroupedQueryAttentionDecodeTraceModule module;
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 4, 1, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 4, 3}, mllm::kFloat32, mllm::kCPU));
  auto ir_op = findGroupedQueryAttentionOp(ir_ctx->topLevelOp());
  ASSERT_NE(ir_op, nullptr);
  ASSERT_NE(ir_op->getAOp(), nullptr);
  EXPECT_EQ(ir_op->getAOp()->getOpType(), mllm::OpTypes::kGroupedQueryAttention);

  const auto options = mllm::jit::binary::dumpLinalgIROptions(ir_op);
  EXPECT_EQ(options.at("implementation"), "DecodeNativeKV");
  const nlohmann::json encoded = {{"op_type", "GroupedQueryAttention"}, {"backend", "CPU"}, {"op_options", options}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kGroupedQueryAttention);
}

// Graphs serialized before the decode-only operation was folded into
// GroupedQueryAttention must still reconstruct.
TEST_F(GroupedQueryAttentionTest, LegacyDecodeOpTypeStringStillReconstructs) {
  const nlohmann::json encoded = {
      {"op_type", "GroupedQueryAttentionDecode"}, {"backend", "CPU"}, {"op_options", nlohmann::json::object()}};
  const auto restored = mllm::jit::interpreter::aopsFromJson(encoded);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->getOpType(), mllm::OpTypes::kGroupedQueryAttention);
}

TEST_F(GroupedQueryAttentionTest, SupportsTransposedNonContiguousHeadViews) {
  auto query_bshd = sequential({1, 3, 4, 5}, 0.07F);
  auto key_bshd = sequential({1, 3, 2, 5}, 0.11F);
  auto value_bshd = sequential({1, 3, 2, 5}, 0.13F);
  auto query = query_bshd.transpose(1, 2);
  auto key = key_bshd.transpose(1, 2);
  auto value = value_bshd.transpose(1, 2);

  const auto actual = mllm::nn::functional::groupedQueryAttention(query, key, value);
  const auto expected = gqaReference(query.contiguous(), key.contiguous(), value.contiguous());
  expectNear(actual, expected);
}
}  // namespace

TEST_F(GroupedQueryAttentionTest, SlidingWindowBoundaryAndChunks) {
  for (int n : {511, 512, 513, 1025}) {
    auto q = sequential({1, 4, n, 8}, 0.023F), k = sequential({1, 2, n, 8}, 0.031F), v = sequential({1, 2, n, 8}, 0.017F);
    GroupedQueryAttentionTraceModule module;
    auto& op = module.op;
    auto actual = op(q, k, v);
    auto expected = gqaReference(q, k, v, 512);
    for (size_t i = 0; i < actual.numel(); ++i) EXPECT_NEAR(actual.ptr<float>()[i], expected.ptr<float>()[i], 2e-5F);
    int pos = 0;
    while (pos < n) {
      int end = std::min(n, pos + 137), begin = std::max(0, pos - 511);
      auto part = op(q[{mllm::kAll, mllm::kAll, {pos, end}, mllm::kAll}], k[{mllm::kAll, mllm::kAll, {begin, end}, mllm::kAll}],
                     v[{mllm::kAll, mllm::kAll, {begin, end}, mllm::kAll}]);
      for (int h = 0; h < 4; ++h)
        for (int t = pos; t < end; ++t)
          for (int d = 0; d < 8; ++d)
            EXPECT_NEAR(part.ptr<float>()[(h * (end - pos) + t - pos) * 8 + d], actual.ptr<float>()[(h * n + t) * 8 + d],
                        2e-5F);
      pos = end;
    }
  }
}
TEST_F(GroupedQueryAttentionTest, InvalidWindow) {
  auto q = sequential({1, 2, 1, 8}, 0.1F);
  GroupedQueryAttentionTraceModule bad_module(mllm::aops::GroupedQueryAttentionImplementation::kDirectStrided, -1);
  auto& bad = bad_module.op;
  EXPECT_THROW(bad(q, q, q), std::invalid_argument);
  GroupedQueryAttentionTraceModule decode_module(mllm::aops::GroupedQueryAttentionImplementation::kDecodeNativeKV, 512);
  auto& decode = decode_module.op;
  EXPECT_THROW(decode(q, q, q), std::invalid_argument);
}
