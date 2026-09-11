// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <cmath>
#include "mllm/mllm.hpp"
#include "mllm/nn/layers/GELU.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
class GELUTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};
class ExactGELUModule : public mllm::nn::Module {
 public:
  ExactGELUModule() : Module("exact_gelu") {
    op = reg<mllm::nn::GELU>("gelu", mllm::aops::GELUOpOptions{.approximate = false});
  }
  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& x, const std::vector<mllm::AnyValue>&) override {
    return {op(x[0])};
  }
  mllm::nn::GELU op;
};
TEST_F(GELUTest, ExactErfAndTail) {
  ExactGELUModule module;
  auto& gelu = module.op;
  auto x = mllm::Tensor::empty({137}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int i = 0; i < 137; ++i) x.ptr<float>()[i] = (i - 68) * 0.1F;
  auto y = gelu(x);
  for (int i = 0; i < 137; ++i) {
    double v = x.ptr<float>()[i];
    EXPECT_NEAR(y.ptr<float>()[i], 0.5 * v * std::erfc(-v / std::sqrt(2.)), 2e-6);
  }
  EXPECT_THROW(gelu(mllm::Tensor::zeros({2}, mllm::kFloat16, mllm::kCPU)), std::invalid_argument);
}
static mllm::ir::linalg::GELUOp::ptr_t findGelu(const mllm::ir::node_ptr_t& n) {
  if (n->isa_<mllm::ir::linalg::GELUOp>()) return n->cast_<mllm::ir::linalg::GELUOp>();
  if (n->isa_<mllm::ir::Op>())
    for (auto& r : n->cast_<mllm::ir::Op>()->regions())
      for (auto& o : r->ops())
        if (auto f = findGelu(o)) return f;
  return nullptr;
}
TEST_F(GELUTest, ExactOptionTraceRoundTrip) {
  ExactGELUModule m;
  auto ir = mllm::ir::trace(m, mllm::Tensor::empty({4}, mllm::kFloat32, mllm::kCPU));
  auto op = findGelu(ir->topLevelOp());
  ASSERT_NE(op, nullptr);
  auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(options.at("approximate"), false);
  auto restored =
      mllm::jit::interpreter::aopsFromJson(nlohmann::json{{"op_type", "GELU"}, {"backend", "CPU"}, {"op_options", options}});
  EXPECT_FALSE(std::static_pointer_cast<mllm::aops::GELUOp>(restored)->options().approximate);
  auto legacy = mllm::jit::interpreter::aopsFromJson(nlohmann::json{{"op_type", "GELU"}, {"backend", "CPU"}});
  EXPECT_TRUE(std::static_pointer_cast<mllm::aops::GELUOp>(legacy)->options().approximate);
}
