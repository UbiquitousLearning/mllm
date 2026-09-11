// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <cmath>
#include "mllm/mllm.hpp"
#include "mllm/nn/layers/Sigmoid.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
class SigmoidTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};
class ExactSigmoidModule : public mllm::nn::Module {
 public:
  ExactSigmoidModule() : Module("exact_sigmoid") {
    op = reg<mllm::nn::Sigmoid>("sigmoid", mllm::aops::SigmoidOpOptions{.approximate = false});
  }
  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& x, const std::vector<mllm::AnyValue>&) override {
    return {op(x[0])};
  }
  mllm::nn::Sigmoid op;
};
TEST_F(SigmoidTest, AccurateLogisticAndTail) {
  ExactSigmoidModule module;
  auto& sigmoid = module.op;
  auto x = mllm::Tensor::empty({137}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int i = 0; i < 137; ++i) x.ptr<float>()[i] = (i - 68) * 0.3F;
  auto y = sigmoid(x);
  for (int i = 0; i < 137; ++i) {
    double v = x.ptr<float>()[i];
    EXPECT_NEAR(y.ptr<float>()[i], 1.0 / (1.0 + std::exp(-v)), 1e-7);
  }
  EXPECT_THROW(sigmoid(mllm::Tensor::zeros({2}, mllm::kFloat16, mllm::kCPU)), std::invalid_argument);
}
static mllm::ir::linalg::SigmoidOp::ptr_t findSigmoid(const mllm::ir::node_ptr_t& n) {
  if (n->isa_<mllm::ir::linalg::SigmoidOp>()) return n->cast_<mllm::ir::linalg::SigmoidOp>();
  if (n->isa_<mllm::ir::Op>())
    for (auto& r : n->cast_<mllm::ir::Op>()->regions())
      for (auto& o : r->ops())
        if (auto f = findSigmoid(o)) return f;
  return nullptr;
}
TEST_F(SigmoidTest, ExactOptionTraceRoundTrip) {
  ExactSigmoidModule m;
  auto ir = mllm::ir::trace(m, mllm::Tensor::empty({4}, mllm::kFloat32, mllm::kCPU));
  auto op = findSigmoid(ir->topLevelOp());
  ASSERT_NE(op, nullptr);
  auto options = mllm::jit::binary::dumpLinalgIROptions(op);
  EXPECT_EQ(options.at("approximate"), false);
  auto restored =
      mllm::jit::interpreter::aopsFromJson(nlohmann::json{{"op_type", "Sigmoid"}, {"backend", "CPU"}, {"op_options", options}});
  EXPECT_FALSE(std::static_pointer_cast<mllm::aops::SigmoidOp>(restored)->options().approximate);
  auto legacy = mllm::jit::interpreter::aopsFromJson(nlohmann::json{{"op_type", "Sigmoid"}, {"backend", "CPU"}});
  EXPECT_TRUE(std::static_pointer_cast<mllm::aops::SigmoidOp>(legacy)->options().approximate);
}
