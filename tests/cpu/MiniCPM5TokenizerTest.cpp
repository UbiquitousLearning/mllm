// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm5/tokenization_minicpm5.hpp"

namespace {
// The runtime context is process-global; tests that allocate tensors share one initialization.
void ensureContext() {
  static const bool initialized = [] {
    mllm::initializeContext();
    return true;
  }();
  (void)initialized;
}
}  // namespace

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

TEST(MiniCPM5TokenizerTest, RendersThroughTheConfiguredChatTemplateBackend) {
  const char* tokenizer_path = std::getenv("MLLM_MINICPM5_TOKENIZER_JSON");
  if (tokenizer_path == nullptr) GTEST_SKIP() << "Set MLLM_MINICPM5_TOKENIZER_JSON for the official-tokenizer oracle";
  ensureContext();

  // Legacy stays byte-identical to the migration formatter.
  mllm::models::minicpm5::MiniCPM5Tokenizer legacy(tokenizer_path);
  EXPECT_EQ(legacy.chatTemplateBackend(), mllm::preprocessor::ChatTemplateBackend::Legacy);
  const mllm::models::minicpm5::MiniCPM5Message message{.prompt = "Hello", .system = "Be concise.", .enable_thinking = true};
  EXPECT_EQ(legacy.renderChatTemplate(message), mllm::models::minicpm5::MiniCPM5Tokenizer::applyChatTemplate(message));

  // jinja_required loads the official template next to tokenizer.json and must
  // reproduce the same runner prompt; without Jinja support it fails closed.
  const mllm::preprocessor::ChatPreprocessorConfig jinja_config{
      .backend = mllm::preprocessor::ChatTemplateBackend::JinjaRequired,
      .template_options = {.model_directory = std::filesystem::path(tokenizer_path).parent_path()}};
  if (!mllm::preprocessor::jinjaChatTemplatesAvailable()) {
    EXPECT_THROW(mllm::models::minicpm5::MiniCPM5Tokenizer(tokenizer_path, jinja_config), mllm::preprocessor::ChatTemplateError);
    return;
  }
  if (!std::filesystem::exists(std::filesystem::path(tokenizer_path).parent_path() / "chat_template.jinja")) {
    GTEST_SKIP() << "tokenizer.json is not inside an official checkpoint directory with chat_template.jinja";
  }
  mllm::models::minicpm5::MiniCPM5Tokenizer jinja(tokenizer_path, jinja_config);
  EXPECT_EQ(jinja.chatTemplateBackend(), mllm::preprocessor::ChatTemplateBackend::JinjaRequired);
  for (const auto& probe : {mllm::models::minicpm5::MiniCPM5Message{.prompt = "Hello"}, message}) {
    EXPECT_EQ(jinja.renderChatTemplate(probe), legacy.renderChatTemplate(probe));
    const auto jinja_ids = jinja.convertMessage(probe).at("sequence");
    const auto legacy_ids = legacy.convertMessage(probe).at("sequence");
    EXPECT_EQ(std::vector<int64_t>(jinja_ids.ptr<int64_t>(), jinja_ids.ptr<int64_t>() + jinja_ids.numel()),
              std::vector<int64_t>(legacy_ids.ptr<int64_t>(), legacy_ids.ptr<int64_t>() + legacy_ids.numel()));
  }
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

  ensureContext();
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
