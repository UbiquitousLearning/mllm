// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/ling3/modeling_ling3.hpp"
#include "mllm/mllm.hpp"

#include <gtest/gtest.h>

#include <cmath>

class Ling3RoPETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

TEST_F(Ling3RoPETest, AppliesOfficialAdjacentPairInterleave) {
  auto input = mllm::Tensor::empty({1, 1, 2, 4}, mllm::kFloat32, mllm::kCPU).alloc();
  auto cos = mllm::Tensor::empty({1, 2, 4}, mllm::kFloat32, mllm::kCPU).alloc();
  auto sin = mllm::Tensor::empty({1, 2, 4}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < 8; ++index) { input.ptr<float>()[index] = static_cast<float>(index + 1); }
  const float angles[] = {0.2F, -0.4F, 0.7F, 0.3F};
  for (int token = 0; token < 2; ++token) {
    for (int pair = 0; pair < 2; ++pair) {
      const float value = angles[token * 2 + pair];
      cos.ptr<float>()[token * 4 + pair] = std::cos(value);
      cos.ptr<float>()[token * 4 + 2 + pair] = std::cos(value);
      sin.ptr<float>()[token * 4 + pair] = std::sin(value);
      sin.ptr<float>()[token * 4 + 2 + pair] = std::sin(value);
    }
  }

  const auto output = mllm::models::ling3::applyLing3InterleavedRoPE(input, cos, sin);
  for (int token = 0; token < 2; ++token) {
    for (int pair = 0; pair < 2; ++pair) {
      const float even = input.ptr<float>()[token * 4 + pair * 2];
      const float odd = input.ptr<float>()[token * 4 + pair * 2 + 1];
      const float angle = angles[token * 2 + pair];
      EXPECT_NEAR(output.ptr<float>()[token * 4 + pair], even * std::cos(angle) - odd * std::sin(angle), 1.0e-6F);
      EXPECT_NEAR(output.ptr<float>()[token * 4 + 2 + pair], odd * std::cos(angle) + even * std::sin(angle), 1.0e-6F);
    }
  }
}

TEST_F(Ling3RoPETest, PadsMLAValuesWithoutChangingPayload) {
  auto input = mllm::Tensor::empty({1, 2, 3, 4}, mllm::kFloat32, mllm::kCPU).alloc();
  for (int index = 0; index < 24; ++index) { input.ptr<float>()[index] = static_cast<float>(index + 1); }
  const auto output = mllm::models::ling3::padLing3ValuesForCache(input, 6);
  ASSERT_EQ(output.shape(), (mllm::Tensor::shape_t{1, 2, 3, 6}));
  for (int vector = 0; vector < 6; ++vector) {
    for (int dim = 0; dim < 4; ++dim) {
      EXPECT_EQ(output.ptr<float>()[vector * 6 + dim], input.ptr<float>()[vector * 4 + dim]);
    }
    EXPECT_EQ(output.ptr<float>()[vector * 6 + 4], 0.0F);
    EXPECT_EQ(output.ptr<float>()[vector * 6 + 5], 0.0F);
  }
}
