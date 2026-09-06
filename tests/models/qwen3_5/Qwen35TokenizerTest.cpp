// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"

#include "mllm/models/qwen3_5/tokenization_qwen3_5.hpp"

TEST(Qwen35TokenizerTest, SplitsContractionsCaseInsensitively) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::qwen3_5::qwen3_5Regex("I'M sure", pieces));

  const std::vector<std::wstring> expected = {L"I", L"'M", L" sure"};
  EXPECT_EQ(pieces, expected);
}

TEST(Qwen35TokenizerTest, PreservesRegexWhitespaceBacktracking) {
  std::vector<std::wstring> pieces;
  ASSERT_TRUE(mllm::models::qwen3_5::qwen3_5Regex("a  b\n  c", pieces));

  const std::vector<std::wstring> expected = {
      L"a", L" ", L" b", L"\n", L" ", L" c",
  };
  EXPECT_EQ(pieces, expected);
}

TEST(Qwen35TokenizerTest, StreamsUtf8SplitAcrossTokens) {
  mllm::models::qwen3_5::Qwen3_5StreamingUtf8Decoder decoder;

  EXPECT_EQ(decoder.append("\xC2"), "");
  EXPECT_EQ(decoder.append("\xA1"), "\xC2\xA1");
  EXPECT_EQ(decoder.finish(), "");
}

TEST(Qwen35TokenizerTest, ReplacesMalformedAndIncompleteUtf8) {
  mllm::models::qwen3_5::Qwen3_5StreamingUtf8Decoder decoder;

  EXPECT_EQ(decoder.append("\xE2("), "\xEF\xBF\xBD(");
  EXPECT_EQ(decoder.append("\xF0\x9F"), "");
  EXPECT_EQ(decoder.finish(), "\xEF\xBF\xBD");
}

TEST(Qwen35TokenizerTest, RendersThroughTheConfiguredChatTemplateBackend) {
  const char* tokenizer_path = std::getenv("MLLM_QWEN35_TOKENIZER_JSON");
  if (tokenizer_path == nullptr) GTEST_SKIP() << "Set MLLM_QWEN35_TOKENIZER_JSON for the official-tokenizer oracle";
  mllm::initializeContext();
  using mllm::models::qwen3_5::Qwen3_5Message;
  using mllm::models::qwen3_5::Qwen3_5Tokenizer;

  // Legacy keeps the pre-Jinja runner prompt bytes.
  Qwen3_5Tokenizer legacy(tokenizer_path);
  EXPECT_EQ(legacy.chatTemplateBackend(), mllm::preprocessor::ChatTemplateBackend::Legacy);
  const Qwen3_5Message message{.prompt = "你好，介绍一下 mllm。"};
  EXPECT_EQ(legacy.renderChatTemplate(message),
            "<|im_start|>user\n你好，介绍一下 mllm。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  EXPECT_EQ(legacy.renderChatTemplate({.prompt = "Describe.", .image_paths = {"a.png", "b.png"}}),
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
            "Describe.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

  // jinja_required loads the official template next to tokenizer.json and must
  // reproduce the same runner prompt; without Jinja support it fails closed.
  const mllm::preprocessor::ChatPreprocessorConfig jinja_config{
      .backend = mllm::preprocessor::ChatTemplateBackend::JinjaRequired,
      .template_options = {.model_directory = std::filesystem::path(tokenizer_path).parent_path()}};
  if (!mllm::preprocessor::jinjaChatTemplatesAvailable()) {
    EXPECT_THROW(Qwen3_5Tokenizer(tokenizer_path, 256 * 256, 512 * 512, 16, 2, 2, 4 * 32 * 32, 24 * 32 * 32 * 1024, jinja_config),
                 mllm::preprocessor::ChatTemplateError);
    return;
  }
  if (!std::filesystem::exists(std::filesystem::path(tokenizer_path).parent_path() / "chat_template.jinja")) {
    GTEST_SKIP() << "tokenizer.json is not inside an official checkpoint directory with chat_template.jinja";
  }
  Qwen3_5Tokenizer jinja(tokenizer_path, 256 * 256, 512 * 512, 16, 2, 2, 4 * 32 * 32, 24 * 32 * 32 * 1024, jinja_config);
  EXPECT_EQ(jinja.chatTemplateBackend(), mllm::preprocessor::ChatTemplateBackend::JinjaRequired);
  for (const auto& probe : {message, Qwen3_5Message{.prompt = "Describe.", .image_paths = {"a.png", "b.png"}}}) {
    EXPECT_EQ(jinja.renderChatTemplate(probe), legacy.renderChatTemplate(probe));
  }
  const auto jinja_ids = jinja.convertMessage(message).at("sequence");
  const auto legacy_ids = legacy.convertMessage(message).at("sequence");
  EXPECT_EQ(std::vector<int64_t>(jinja_ids.ptr<int64_t>(), jinja_ids.ptr<int64_t>() + jinja_ids.numel()),
            std::vector<int64_t>(legacy_ids.ptr<int64_t>(), legacy_ids.ptr<int64_t>() + legacy_ids.numel()));
}
