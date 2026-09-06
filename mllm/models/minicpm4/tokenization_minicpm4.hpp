// tokenization_minicpm4.hpp
#pragma once

#include <filesystem>
#include <utility>
#include <vector>
#include <unordered_map>
#include "mllm/preprocessor/tokenizers/BPEUTF8.hpp"  //BPEUTF8, MiniCPM4 use LlamaTokenizer!
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/minicpm4/configuration_minicpm4.hpp"
#include "mllm/preprocessor/chat_template/LegacyChatMl.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::minicpm4 {

struct MiniCPM4Message {
  std::string prompt;
};

class MiniCPM4Tokenizer final : public mllm::preprocessor::AutoTokenizerUTF8 {
 public:
  // The chat-template backend is selected explicitly; see Qwen3Tokenizer.
  explicit MiniCPM4Tokenizer(const std::string& file_path, preprocessor::ChatPreprocessorConfig chat_template = {})
      : chat_preprocessor_(std::move(chat_template), preprocessor::renderLegacyChatMlSingleTurn) {
    bpe_.initFromSentencePieceJson(file_path);
    chat_preprocessor_.setControlTokens(bpe_.controlTokens());

    special_tokens_trie_.add("<|im_start|>");
    special_tokens_trie_.add("<|im_end|>");
    special_tokens_trie_.add("<|endoftext|>");
    special_tokens_trie_.add("<|tool_call|>");
    special_tokens_trie_.add("<|execute_start|>");
    special_tokens_trie_.add("<|execute_end|>");
    special_tokens_trie_.add("<|fim_prefix|>");
    special_tokens_trie_.add("<|fim_middle|>");
    special_tokens_trie_.add("<|fim_suffix|>");
  }

  std::string normalize(const std::string& text) {
    std::string normalized;

    // add ▁
    normalized += "\xE2\x96\x81";

    // replace space with ▁
    for (char c : text) {
      if (c == ' ') {
        normalized += "\xE2\x96\x81";  // ▁
      } else {
        normalized += c;
      }
    }

    return normalized;
  }

  std::vector<int64_t> encode(const std::string& str) override {
    auto tokens = tokenize(str);
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) { ids.push_back(bpe_._lookup_vocab(token)); }
    return ids;
  }

  std::string decode(const std::vector<int64_t>& ids) override {
    std::string result;
    for (auto id : ids) { result += bpe_._lookup_inverse_vocab(id); }

    // remove _
    if (result.size() >= 3 && result[0] == '\xE2' && result[1] == '\x96' && result[2] == '\x81') { result = result.substr(3); }

    // replace
    std::string decoded;
    for (size_t i = 0; i < result.size();) {
      if (i + 2 < result.size() && result[i] == '\xE2' && result[i + 1] == '\x96' && result[i + 2] == '\x81') {
        decoded += ' ';
        i += 3;
      } else {
        decoded += result[i];
        i++;
      }
    }

    return decoded;
  }

  std::vector<std::string> tokenize(const std::string& str) override {
    auto segments = special_tokens_trie_.split(str);

    std::vector<std::string> all_tokens;
    for (const auto& segment : segments) {
      if (special_tokens_trie_.isSpecialToken(segment)) {
        all_tokens.push_back(segment);
        continue;
      }

      std::string normalized = normalize(segment);

      auto bpe_tokens = bpe_._bpe(normalized);
      all_tokens.insert(all_tokens.end(), bpe_tokens.begin(), bpe_tokens.end());
    }

    return all_tokens;
  }

  std::string detokenize(int64_t token_id) {
    auto token = bpe_._lookup_inverse_vocab(token_id);
    return decode_token(token);
  }

  std::string detokenize(const std::vector<std::string>& tokens) override {
    std::string result;
    for (const auto& token : tokens) { result += token; }
    return decode_string(result);
  }

  MiniCPM4Tokenizer(const std::string& file_path, const MiniCPM4Config& config, std::filesystem::path model_directory = {})
      : MiniCPM4Tokenizer(file_path,
                          preprocessor::ChatPreprocessorConfig{
                              .backend = config.chat_template_backend,
                              .template_options = {.model_directory = model_directory.empty()
                                                                          ? std::filesystem::path(file_path).parent_path()
                                                                          : std::move(model_directory)}}) {}

  preprocessor::ChatTemplateBackend chatTemplateBackend() const noexcept { return chat_preprocessor_.backend(); }

  // Renders the single-turn runner prompt through the configured backend.
  std::string renderChatTemplate(const MiniCPM4Message& message) const {
    return chat_preprocessor_.render(preprocessor::makeSingleTurnChatTemplateRequest(message.prompt, "", std::nullopt));
  }

  ARGenerationOutputPast convertMessage(const MiniCPM4Message& message) {
    auto applied_string = renderChatTemplate(message);

    auto tokens = tokenize(applied_string);

    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) { ids.push_back(bpe_._lookup_vocab(token)); }

    Tensor sequence =
        Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kNormal).setName("minicpm4-tokenizer-i0").alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {{"sequence", sequence}};
  }

 private:
  preprocessor::ChatPreprocessor chat_preprocessor_;
  preprocessor::BPEUTF8 bpe_;

  std::string decode_token(const std::string& token) {
    std::string decoded;

    // replace _ with space
    for (size_t i = 0; i < token.size();) {
      if (i + 2 < token.size() && token[i] == '\xE2' && token[i + 1] == '\x96' && token[i + 2] == '\x81') {
        decoded += ' ';
        i += 3;
      } else {
        decoded += token[i];
        i++;
      }
    }

    return decoded;
  }

  std::string decode_string(const std::string& text) {
    // remove _
    std::string result = text;
    if (result.size() >= 3 && result[0] == '\xE2' && result[1] == '\x96' && result[2] == '\x81') { result = result.substr(3); }

    // 2. replace all _ with space
    std::string decoded;
    for (size_t i = 0; i < result.size();) {
      if (i + 2 < result.size() && result[i] == '\xE2' && result[i + 1] == '\x96' && result[i + 2] == '\x81') {
        decoded += ' ';
        i += 3;
      } else {
        decoded += result[i];
        i++;
      }
    }

    return decoded;
  }
};

}  // namespace mllm::models::minicpm4
