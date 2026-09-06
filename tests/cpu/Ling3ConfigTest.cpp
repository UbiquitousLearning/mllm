// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/ling3/configuration_ling3.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace {

std::string ling3RuntimeConfigPath() {
  const char* value = std::getenv("LING3_RUNTIME_CONFIG");
  return value == nullptr ? std::string(LING3_EXAMPLE_DIR) + "/config_tiny_w4a32_kai.json" : std::string(value);
}

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
