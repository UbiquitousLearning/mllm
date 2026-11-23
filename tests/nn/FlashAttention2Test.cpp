// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"

namespace {

using mllm::Tensor;
using mllm::aops::FlashAttention2Layout;
using mllm::nn::functional::flashAttention2;

class FlashAttention2Test : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

class FlashAttention2TraceModule final : public mllm::nn::Module {
 public:
  FlashAttention2TraceModule() : Module("fa2_trace") {}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<mllm::AnyValue>&) override {
    return {flashAttention2(inputs[0], inputs[1], inputs[2], FlashAttention2Layout::kBHSD, true, 7)};
  }
};

mllm::ir::linalg::FlashAttention2Op::ptr_t findAttention(const mllm::ir::node_ptr_t& node) {
  if (node->isa_<mllm::ir::linalg::FlashAttention2Op>()) return node->cast_<mllm::ir::linalg::FlashAttention2Op>();
  if (!node->isa_<mllm::ir::Op>()) return nullptr;
  for (const auto& region : node->cast_<mllm::ir::Op>()->regions()) {
    for (const auto& op : region->ops()) {
      if (auto found = findAttention(op)) return found;
    }
  }
  return nullptr;
}

Tensor pattern(const Tensor::shape_t& shape, float phase) {
  auto tensor = Tensor::empty(shape, mllm::kFloat32, mllm::kCPU).alloc();
  for (int64_t i = 0; i < tensor.numel(); ++i) tensor.ptr<float>()[i] = std::sin(phase * static_cast<float>(i + 1));
  return tensor;
}

TEST_F(FlashAttention2Test, CachedWindowMatchesOneShotWithNativeKvHeads) {
  const auto q = pattern({2, 23, 8, 19}, 0.07f).transpose(1, 2);
  const auto key_cache = pattern({2, 2, 41, 19}, 0.19f);
  const auto value_cache = pattern({2, 2, 41, 5}, 0.11f);
  const auto k = key_cache[{mllm::kAll, mllm::kAll, {0, 23}, mllm::kAll}];
  const auto v = value_cache[{mllm::kAll, mllm::kAll, {0, 23}, mllm::kAll}];
  const auto suffix = q[{mllm::kAll, mllm::kAll, {17, 23}, mllm::kAll}];
  const auto full = flashAttention2(q, k, v, FlashAttention2Layout::kBHSD, true, 7);
  const auto chunk = flashAttention2(suffix, k, v, FlashAttention2Layout::kBHSD, true, 7);
  ASSERT_EQ(chunk.shape(), (Tensor::shape_t{2, 8, 6, 5}));
  ASSERT_GT(k.stride()[1], k.shape()[2] * k.shape()[3]);
  const auto q_len = chunk.shape()[2];
  for (int b = 0; b < 2; ++b) {
    for (int h = 0; h < 8; ++h) {
      for (int s = 0; s < q_len; ++s) {
        for (int d = 0; d < 5; ++d) {
          EXPECT_FLOAT_EQ(*chunk.cptrAt<float>({b, h, s, d}), *full.cptrAt<float>({b, h, 17 + s, d}));
        }
      }
    }
  }
}

TEST_F(FlashAttention2Test, LegacyEntryStillReadsPaddedBshdCacheViews) {
  const auto q = pattern({2, 5, 4, 8}, 0.07f);
  const auto key_cache = pattern({2, 41, 2, 8}, 0.19f);
  const auto value_cache = pattern({2, 41, 2, 8}, 0.11f);
  auto k = key_cache[{mllm::kAll, {0, 11}, mllm::kAll, mllm::kAll}];
  auto v = value_cache[{mllm::kAll, {0, 11}, mllm::kAll, mllm::kAll}];
  ASSERT_GT(k.stride()[0], k.shape()[1] * k.shape()[2] * k.shape()[3]);
  const auto actual = flashAttention2(q, k, v);
  const auto dense = flashAttention2(q, k.contiguous(), v.contiguous());
  ASSERT_EQ(actual.shape(), dense.shape());
  for (int64_t i = 0; i < actual.numel(); ++i) { EXPECT_NEAR(actual.ptr<float>()[i], dense.ptr<float>()[i], 1e-4f); }
}

TEST_F(FlashAttention2Test, ValidatesLayoutAndGeometry) {
  const auto q = pattern({1, 3, 2, 8}, 0.07f);
  const auto k = pattern({1, 7, 2, 8}, 0.11f);
  const auto short_k = pattern({1, 2, 2, 8}, 0.13f);
  EXPECT_THROW(flashAttention2(q, k, k, FlashAttention2Layout::kBSHD, false, 2), std::invalid_argument);
  EXPECT_THROW(flashAttention2(q, short_k, short_k), std::invalid_argument);
  EXPECT_THROW(flashAttention2(q, k, k, static_cast<FlashAttention2Layout>(99)), std::invalid_argument);
  EXPECT_EQ(flashAttention2(q, short_k, short_k, FlashAttention2Layout::kBSHD, false).shape(), q.shape());
}

TEST_F(FlashAttention2Test, TracePreservesLayoutAndCausalWindow) {
  FlashAttention2TraceModule module;
  auto ir_ctx = mllm::ir::trace(module, Tensor::empty({1, 4, 2, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 9, 5}, mllm::kFloat32, mllm::kCPU),
                                Tensor::empty({1, 2, 9, 3}, mllm::kFloat32, mllm::kCPU));
  const auto op = findAttention(ir_ctx->topLevelOp());
  ASSERT_NE(op, nullptr);
  const auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  const auto restored =
      mllm::jit::interpreter::aopsFromJson({{"op_type", "FlashAttention2"}, {"backend", "CPU"}, {"op_options", options}});
  const auto fa2 = std::dynamic_pointer_cast<mllm::aops::FlashAttention2Op>(restored);
  ASSERT_NE(fa2, nullptr);
  EXPECT_EQ(fa2->options().layout, FlashAttention2Layout::kBHSD);
  EXPECT_EQ(fa2->options().sliding_window, 7);
  EXPECT_TRUE(fa2->options().causal_mask);
  EXPECT_TRUE(fa2->options().use_tiled);
}

TEST_F(FlashAttention2Test, LegacyGraphsDefaultToCausalBshd) {
  const auto restored =
      mllm::jit::interpreter::aopsFromJson({{"op_type", "FlashAttention2"},
                                            {"backend", "CPU"},
                                            {"op_options", {{"B", 1}, {"q_head", 4}, {"kv_head", 2}, {"D", 8}}}});
  const auto fa2 = std::dynamic_pointer_cast<mllm::aops::FlashAttention2Op>(restored);
  ASSERT_NE(fa2, nullptr);
  EXPECT_EQ(fa2->options().layout, FlashAttention2Layout::kBSHD);
  EXPECT_EQ(fa2->options().sliding_window, 0);
  EXPECT_TRUE(fa2->options().causal_mask);
  EXPECT_FALSE(fa2->options().use_tiled);
}

}  // namespace
