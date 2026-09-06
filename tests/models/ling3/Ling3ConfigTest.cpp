// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/ling3/configuration_ling3.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace {

auto exampleDir() -> std::string {
  const char* example_dir_override = std::getenv("MLLM_LING3_EXAMPLE_DIR");
  return example_dir_override == nullptr ? std::string(LING3_EXAMPLE_DIR) : std::string(example_dir_override);
}

std::string ling3RuntimeConfigPath() { return exampleDir() + "/config_tiny_w4a32_kai.json"; }

}  // namespace

TEST(Ling3Config, AcceptsPinnedTinyRuntimeContract) {
  const mllm::models::ling3::Ling3Config config(ling3RuntimeConfigPath());
  EXPECT_TRUE(mllm::models::ling3::hasOfficialLing3TinyArchitecture(config));
  EXPECT_EQ(config.numFullAttentionLayers(), 6);
  EXPECT_EQ(config.numKDALayers(), 18);
  for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
    EXPECT_EQ(config.isFullAttentionLayer(layer), (layer + 1) % 4 == 0);
  }
}
