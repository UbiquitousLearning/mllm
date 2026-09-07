// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "mllm/models/minicpm5/configuration_minicpm5.hpp"

namespace {

auto exampleDir() -> std::string {
  const char* example_dir_override = std::getenv("MLLM_MINICPM5_EXAMPLE_DIR");
  return example_dir_override == nullptr ? std::string(MINICPM5_EXAMPLE_DIR) : std::string(example_dir_override);
}

auto loadConfig() -> mllm::models::minicpm5::MiniCPM5Config {
  return mllm::models::minicpm5::MiniCPM5Config(exampleDir() + "/config_1B_w4a32_kai.json");
}

auto parameterFile(mllm::ModelFileVersion version, const std::vector<int32_t>& embedding_shape, bool include_lm_head = true)
    -> mllm::ParameterFile::ptr_t {
  auto parameters = mllm::ParameterFile::create(version);
  auto embedding = mllm::Tensor::empty(embedding_shape, mllm::kFloat32, mllm::kCPU);
  embedding.impl()->storage()->mem_type_ = mllm::kManual;
  parameters->push("model.embed_tokens.weight", embedding);
  if (include_lm_head) {
    auto lm_head = mllm::Tensor::empty({1}, mllm::kFloat32, mllm::kCPU);
    lm_head.impl()->storage()->mem_type_ = mllm::kManual;
    parameters->push("lm_head.weight", lm_head);
  }
  return parameters;
}

}  // namespace

TEST(MiniCPM5ConfigTest, PinsOfficialOneBillionParameterContract) {
  const auto config = loadConfig();
  EXPECT_TRUE(mllm::models::minicpm5::matchesOfficialMiniCPM5_1BRuntimeContract(config));
  EXPECT_EQ(config.hidden_size, 1536);
  EXPECT_EQ(config.head_dim, 128);
  EXPECT_EQ(config.num_attention_heads, 16);
  EXPECT_EQ(config.num_key_value_heads, 2);
  EXPECT_EQ(config.num_attention_heads * config.head_dim, 2048);
  EXPECT_EQ(config.num_key_value_heads * config.head_dim, 256);
  EXPECT_EQ(config.eos_token_ids, (std::vector<int64_t>{1, 130073}));
  EXPECT_EQ(config.max_cache_length, 2048);
}

TEST(MiniCPM5ConfigTest, RejectsRuntimeContractDrift) {
  auto config = loadConfig();
  config.head_dim = config.hidden_size / config.num_attention_heads;
  EXPECT_FALSE(mllm::models::minicpm5::matchesOfficialMiniCPM5_1BRuntimeContract(config));
}

TEST(MiniCPM5ConfigTest, ValidatesEmbeddingAndIndependentLmHead) {
  const auto config = loadConfig();
  const auto matching_v2 = parameterFile(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size});
  EXPECT_NO_THROW(mllm::models::minicpm5::validateModelConfigMatch(config, matching_v2));

  const auto wrong_v2 = parameterFile(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size + 1});
  EXPECT_THROW(mllm::models::minicpm5::validateModelConfigMatch(config, wrong_v2), std::invalid_argument);

  const auto flat_elements = static_cast<int32_t>(static_cast<size_t>(config.vocab_size) * config.hidden_size);
  const auto matching_v1 = parameterFile(mllm::ModelFileVersion::kV1, {flat_elements});
  EXPECT_NO_THROW(mllm::models::minicpm5::validateModelConfigMatch(config, matching_v1));

  const auto missing_lm_head = parameterFile(mllm::ModelFileVersion::kV2, {config.vocab_size, config.hidden_size}, false);
  EXPECT_THROW(mllm::models::minicpm5::validateModelConfigMatch(config, missing_lm_head), std::invalid_argument);
}
