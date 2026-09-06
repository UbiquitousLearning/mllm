// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/models/ling3/tokenization_ling3.hpp"
#include "mllm/mllm.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::string officialTokenizerPath() {
  const char* value = std::getenv("LING3_OFFICIAL_TOKENIZER");
  return value == nullptr ? std::string() : std::string(value);
}

std::vector<int64_t> ids(mllm::models::ling3::Ling3Tokenizer& tokenizer, const std::string& text) {
  const auto tensor = tokenizer.convert2Ids(tokenizer.tokenize(text));
  return {tensor.ptr<int64_t>(), tensor.ptr<int64_t>() + tensor.numel()};
}

class Ling3TokenizerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { mllm::initializeContext(); }
};

TEST_F(Ling3TokenizerTest, MatchesOfficialByteBPEAndNFCVectors) {
  const auto path = officialTokenizerPath();
  if (path.empty()) { GTEST_SKIP() << "LING3_OFFICIAL_TOKENIZER is not set"; }
  mllm::models::ling3::Ling3Tokenizer tokenizer(path);
  EXPECT_EQ(ids(tokenizer, "Hello, world!"), (std::vector<int64_t>{14455, 11, 1931, 0}));
  EXPECT_EQ(ids(tokenizer, "你好，世界！"), (std::vector<int64_t>{34355, 44291, 859}));
  EXPECT_EQ(ids(tokenizer, "e\xCC\x81 café"), (std::vector<int64_t>{2900, 67656}));
}

TEST_F(Ling3TokenizerTest, RendersOfficialSingleTurnThinkingTemplates) {
  const auto path = officialTokenizerPath();
  if (path.empty()) { GTEST_SKIP() << "LING3_OFFICIAL_TOKENIZER is not set"; }
  mllm::models::ling3::Ling3Tokenizer tokenizer(path);
  auto enabled = tokenizer.convertMessage({"你好", "", true}).at("sequence");
  const std::vector<int64_t> enabled_expected = {157151, 90827, 157152, 14136,  5381, 6350, 366,  156895, 157151, 39,    116171,
                                                 157152, 34355, 156895, 157151, 8469, 7342, 5468, 157152, 198,    156903};
  EXPECT_EQ(std::vector<int64_t>(enabled.ptr<int64_t>(), enabled.ptr<int64_t>() + enabled.numel()), enabled_expected);

  auto disabled = tokenizer.convertMessage({"Hello", "", false}).at("sequence");
  const std::vector<int64_t> disabled_expected = {157151, 90827, 157152, 14136,  5381,   6350,   928,    156895,
                                                  157151, 39,    116171, 157152, 14455,  156895, 157151, 8469,
                                                  7342,   5468,  157152, 198,    156903, 156904};
  EXPECT_EQ(std::vector<int64_t>(disabled.ptr<int64_t>(), disabled.ptr<int64_t>() + disabled.numel()), disabled_expected);
}

}  // namespace
