// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm5/modeling_minicpm5.hpp"

namespace {

auto exampleDir() -> std::string {
  const char* example_dir_override = std::getenv("MLLM_MINICPM5_EXAMPLE_DIR");
  return example_dir_override == nullptr ? std::string(MINICPM5_EXAMPLE_DIR) : std::string(example_dir_override);
}

}  // namespace

TEST(MiniCPM5ModelTest, BuildsNativeKVHeadLogicalSlotsAndResetsThem) {
  mllm::initializeContext();
  const auto config = mllm::models::minicpm5::MiniCPM5Config(exampleDir() + "/config_1B_w4a32_kai.json");
  auto model = mllm::models::minicpm5::MiniCPM5ForCausalLM(config);
  auto& cache = model.kvCache();

  EXPECT_EQ(cache.getLayerNums(), 24);
  EXPECT_EQ(cache.kvHeads(), 2);
  EXPECT_EQ(cache.headDim(), 128);
  EXPECT_EQ(cache.maxCacheLength(), 2048);
  EXPECT_EQ(cache.getKCacheBuffer(0).shape(), (mllm::Tensor::shape_t{1, 2, 2048, 128}));

  auto key = mllm::Tensor::zeros({1, 2, 1, 128}, mllm::kFloat32, mllm::kCPU);
  auto value = mllm::Tensor::zeros({1, 2, 1, 128}, mllm::kFloat32, mllm::kCPU);
  cache.updateKVCache(7, key, value);
  EXPECT_EQ(cache.getCurrentSeqCnt(7), 1);
  EXPECT_EQ(cache.getCurrentSeqCnt(6), 0);
  model.resetState();
  EXPECT_EQ(cache.getCurrentSeqCnt(7), 0);
}
