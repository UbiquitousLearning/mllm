// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <vector>
#include <unordered_map>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp"
#include "mllm/models/qwen2_5omni/audio_preprocessor_qwen2_5omni.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::models::qwen2_5omni {

// same regex as Qwen2/Qwen2-VL tokenizers
inline bool qwen2_5OmniTokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  {
    size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();

    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }

    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    } else if (has_prefix) {
      pos = original_pos;
      matched.clear();
    }
  }

  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  {
    size_t original_pos = pos;
    matched.clear();
    size_t start = pos;

    if (str[pos] == L' ') { ++pos; }

    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));

      matched = str.substr(start, pos - start);

      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    } else {
      pos = original_pos;
    }
  }

  {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline bool qwen2_5OmniRegex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (qwen2_5OmniTokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

struct Qwen2_5OmniMessage {
  std::string prompt;
  std::string system_prompt = "You are a helpful assistant.";

  [[nodiscard]] std::string buildChatMessage() const {
    std::string result;
    if (!system_prompt.empty()) {
      result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    result += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    return result;
  }
};

struct Qwen2_5OmniVisionMessage {
  std::string prompt;
  std::string img_file_path;
  std::string system_prompt = "You are a helpful assistant.";

  [[nodiscard]] std::string buildChatMessage() const {
    std::string result;
    if (!system_prompt.empty()) {
      result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    result += "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>" + prompt + "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    return result;
  }
};

struct Qwen2_5OmniAudioMessage {
  std::string prompt;
  std::string audio_file_path;
  std::string system_prompt = "You are a helpful assistant.";

  [[nodiscard]] std::string buildChatMessage() const {
    std::string result;
    if (!system_prompt.empty()) {
      result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    result += "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>" + prompt + "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    return result;
  }
};

class Qwen2_5OmniTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit Qwen2_5OmniTokenizer(const std::string& file_path,
                                int32_t spatial_merge_size = 2,
                                int32_t min_pixels = 56 * 56,
                                int32_t max_pixels = 1280 * 1280,
                                int32_t audio_sample_rate = 16000,
                                int32_t audio_n_mels = 128,
                                int32_t audio_hop_length = 160,
                                int32_t audio_chunk_length = 300)
                                //interestingly, the answer went bad when setting max_pixels higher, eg. 3584*3584)
      : image_preprocessor_(min_pixels, max_pixels),
        audio_preprocessor_(audio_sample_rate, audio_n_mels, audio_hop_length, audio_chunk_length),
        spatial_merge_size_(spatial_merge_size) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|object_ref_start|>");
    special_tokens_trie_.add(L"<|object_ref_end|>");
    special_tokens_trie_.add(L"<|box_start|>");
    special_tokens_trie_.add(L"<|box_end|>");
    special_tokens_trie_.add(L"<|quad_start|>");
    special_tokens_trie_.add(L"<|quad_end|>");
    special_tokens_trie_.add(L"<|vision_bos|>");
    special_tokens_trie_.add(L"<|vision_eos|>");
    special_tokens_trie_.add(L"<|vision_pad|>");
    special_tokens_trie_.add(L"<|image_pad|>");
    special_tokens_trie_.add(L"<|video_pad|>");
    special_tokens_trie_.add(L"<|AUDIO|>");
    special_tokens_trie_.add(L"<|audio_bos|>");
    special_tokens_trie_.add(L"<|audio_eos|>");
    special_tokens_trie_.add(L"<|IMAGE|>");
    special_tokens_trie_.add(L"<|VIDEO|>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::qwen2_5omni::qwen2_5OmniRegex(str, splitted);
    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

      auto bpe_ts = bpe_._bpe(mapped_str);

      for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
    }

    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;
    for (const auto& token : tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        all_tokens.emplace_back(token);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  std::wstring detokenize(int64_t pos_idx) override {
    auto str = _detokenize(pos_idx);
    std::string utf_8_str;
    for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
    return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("qwen2_5omni-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return ret;
  }

  ARGenerationOutputPast convertMessage(const Qwen2_5OmniMessage& message) {
    auto applied_string = message.buildChatMessage();
    auto sequence_str = tokenize(applied_string);

    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen2_5omni-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {{"sequence", sequence}};
  }

  ARGenerationOutputPast convertVisionMessage(const Qwen2_5OmniVisionMessage& message) {
    auto applied_string = message.buildChatMessage();

    auto [img, grid_thw] = image_preprocessor_(message.img_file_path);

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    auto grid_t = grid_thw.ptr<int32_t>()[0];
    auto grid_h = grid_thw.ptr<int32_t>()[1];
    auto grid_w = grid_thw.ptr<int32_t>()[2];
    int32_t img_token_nums = grid_t * grid_h * grid_w;
    img_token_nums /= (spatial_merge_size_ * spatial_merge_size_);

    auto image_token_id = bpe_._lookup_vocab(L"<|IMAGE|>");
    {
      auto it = std::find(ids.begin(), ids.end(), image_token_id);
      if (it == ids.end()) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing <|IMAGE|> token in Qwen2.5-Omni prompt template.");
      }
      ids.insert(it + 1, img_token_nums - 1, image_token_id);
    }

    Tensor sequence = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen2_5omni-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
        {"img", img},
        {"grid_thw", grid_thw},
    };
  }

  ARGenerationOutputPast convertAudioMessage(const Qwen2_5OmniAudioMessage& message) {
    auto applied_string = message.buildChatMessage();
    auto sequence_str = tokenize(applied_string);

    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    auto audio_result = audio_preprocessor_.processAudioFile(message.audio_file_path);
    if (audio_result.input_features.isNil() || audio_result.feature_length <= 0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to extract audio features for Qwen2.5-Omni.");
    }

    int32_t audio_token_nums = audio_preprocessor_.calcAudioTokenLength(audio_result.feature_length);
    if (audio_token_nums <= 0) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Invalid audio token length for Qwen2.5-Omni.");
    }

    auto audio_token_id = bpe_._lookup_vocab(L"<|AUDIO|>");
    {
      auto it = std::find(ids.begin(), ids.end(), audio_token_id);
      if (it == ids.end()) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing <|AUDIO|> token in Qwen2.5-Omni prompt template.");
      }
      ids.insert(it + 1, audio_token_nums - 1, audio_token_id);
    }

    Tensor sequence = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen2_5omni-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    audio_result.input_features.setName("input_features");

    return {
        {"sequence", sequence},
        {"input_features", audio_result.input_features},
    };
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
  mllm::models::qwen2vl::Qwen2VLImagePreprocessor image_preprocessor_;
  Qwen2_5OmniAudioPreprocessor audio_preprocessor_;
  int32_t spatial_merge_size_ = 2;
};

}  // namespace mllm::models::qwen2_5omni
