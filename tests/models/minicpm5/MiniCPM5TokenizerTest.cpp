// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm5/tokenization_minicpm5.hpp"

TEST(MiniCPM5TokenizerTest, SplitsDigitsIntoAtMostThreeCharacterChunks) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::minicpm5::miniCPM5Regex("1234567", pieces));
  EXPECT_EQ(pieces, (std::vector<std::wstring>{L"123", L"456", L"7"}));
}

TEST(MiniCPM5TokenizerTest, AppliesOfficialNoToolTemplates) {
  using mllm::models::minicpm5::MiniCPM5Message;
  using mllm::models::minicpm5::MiniCPM5Tokenizer;
  EXPECT_EQ(MiniCPM5Tokenizer::applyChatTemplate({.prompt = "Hello"}),
            "<s><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  EXPECT_EQ(MiniCPM5Tokenizer::applyChatTemplate({.prompt = "Hello", .system = "Be concise.", .enable_thinking = true}),
            "<s><|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n");
}

TEST(MiniCPM5TokenizerTest, StreamsUtf8AcrossTokenBoundaries) {
  mllm::models::minicpm5::MiniCPM5StreamingUtf8Decoder decoder;
  EXPECT_EQ(decoder.append("\xF0\x9F"), "");
  EXPECT_EQ(decoder.append("\x98\x80"), "\xF0\x9F\x98\x80");
  EXPECT_EQ(decoder.finish(), "");
}

TEST(MiniCPM5TokenizerTest, MatchesPinnedOfficialTokenizerWhenProvided) {
  const char* tokenizer_path = std::getenv("MLLM_MINICPM5_TOKENIZER_JSON");
  if (tokenizer_path == nullptr) GTEST_SKIP() << "Set MLLM_MINICPM5_TOKENIZER_JSON for the official-tokenizer oracle";

  mllm::initializeContext();
  mllm::models::minicpm5::MiniCPM5Tokenizer tokenizer(tokenizer_path);
  const auto input = tokenizer.convertMessage({.prompt = "Hello"});
  const auto sequence = input.at("sequence");
  const std::vector<int64_t> expected = {
      0, 130072, 8448, 220, 36417, 130073, 220, 130072, 130071, 220, 8, 130063, 9, 130063,
  };
  const std::vector<int64_t> actual(sequence.ptr<int64_t>(), sequence.ptr<int64_t>() + sequence.numel());
  EXPECT_EQ(actual, expected);

  const auto thinking_input = tokenizer.convertMessage({
      .prompt = "Hello",
      .system = "Be concise.",
      .enable_thinking = true,
  });
  const auto thinking_sequence = thinking_input.at("sequence");
  const std::vector<int64_t> expected_thinking = {
      0, 130072, 17261, 220, 5100, 44375, 35, 130073, 220, 130072, 8448, 220, 36417, 130073, 220, 130072, 130071, 220, 8, 220,
  };
  EXPECT_EQ(
      std::vector<int64_t>(thinking_sequence.ptr<int64_t>(), thinking_sequence.ptr<int64_t>() + thinking_sequence.numel()),
      expected_thinking);

  const auto digits = tokenizer.convert2Ids(tokenizer.tokenize("1234567"));
  EXPECT_EQ(std::vector<int64_t>(digits.ptr<int64_t>(), digits.ptr<int64_t>() + digits.numel()),
            (std::vector<int64_t>{5645, 12740, 44}));

  const char* demo_prompt_path = std::getenv("MLLM_MINICPM5_DEMO_PROMPT_FILE");
  if (demo_prompt_path != nullptr) {
    std::ifstream prompt_stream(demo_prompt_path, std::ios::binary);
    ASSERT_TRUE(prompt_stream) << "unable to read MiniCPM5 demo prompt";
    std::string prompt;
    prompt.assign(std::istreambuf_iterator<char>(prompt_stream), std::istreambuf_iterator<char>());
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r')) prompt.pop_back();
    ASSERT_FALSE(prompt.empty());
    const auto demo_input = tokenizer.convertMessage({.prompt = prompt});
    EXPECT_EQ(demo_input.at("sequence").shape()[1], 200);
  }
}
