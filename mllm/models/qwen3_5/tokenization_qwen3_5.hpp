// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cwctype>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3_5/chat_template_qwen3_5.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
#include "mllm/models/qwen3_5/image_preprocessor_qwen3_5.hpp"
#include "mllm/models/qwen3_5/multimodal_qwen3_5.hpp"
#include "mllm/models/qwen3_5/video_preprocessor_qwen3_5.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::qwen3_5 {

using Qwen3_5StreamingUtf8Decoder = preprocessor::StreamingUtf8Decoder;

// Reuse the Qwen3 regex pattern — same BPE tokenization scheme.
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
inline bool qwen3_5TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    bool matches = pos + contraction.size() <= str.size();
    for (size_t index = 0; matches && index < contraction.size(); ++index) {
      matches = std::towlower(str[pos + index]) == contraction[index];
    }
    if (matches) {
      matched = str.substr(pos, contraction.size());
      pos += contraction.size();
      return true;
    }
  }

  // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+
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
    } else {
      if (has_prefix) {
        pos = original_pos;
        matched.clear();
      }
    }
  }

  // 3. Match \p{N}
  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  // 4. Match ?[^\s\p{L}\p{N}]+[\r\n]*
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

  // 5. Match \s*[\r\n]+
  {
    size_t start = pos;
    size_t scan = pos;
    size_t last_line_break = std::wstring::npos;
    while (scan < str.size() && std::iswspace(str[scan])) {
      if (str[scan] == L'\r' || str[scan] == L'\n') { last_line_break = scan + 1; }
      ++scan;
    }
    if (last_line_break != std::wstring::npos) {
      pos = last_line_break;
      matched = str.substr(start, last_line_break - start);
      return true;
    }
  }

  // 6. Match \s+(?!\S)
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos >= str.size()) {
      matched = str.substr(start, pos - start);
      return true;
    }
    // When followed by a non-space, regex backtracking leaves the final
    // whitespace character for the next alternative.
    if (pos - start > 1) {
      --pos;
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  // 7. Match remaining whitespace
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline bool qwen3_5Regex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (qwen3_5TokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      splitted.push_back(w_string.substr(pos, 1));
      ++pos;
    }
  }
  return true;
}

struct Qwen3_5Message {
  std::string prompt;
  std::vector<std::string> image_paths;
  Tensor video_frames_thwc = Tensor::nil();
  std::vector<int32_t> video_frame_indices;
  double video_frames_per_second = 0.0;
};

class Qwen3_5Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  // The chat-template backend is selected explicitly. The default keeps the
  // migration (legacy) renderer; a JinjaRequired configuration loads the
  // official template from its model directory and fails here when it cannot.
  explicit Qwen3_5Tokenizer(const std::string& file_path, int32_t image_min_pixels = 256 * 256,
                            int32_t image_max_pixels = 512 * 512, int32_t vision_patch_size = 16,
                            int32_t vision_temporal_patch_size = 2, int32_t vision_spatial_merge_size = 2,
                            int32_t video_min_pixels = 4 * 32 * 32, int32_t video_max_pixels = 24 * 32 * 32 * 1024,
                            preprocessor::ChatPreprocessorConfig chat_template = {})
      : image_preprocessor_(image_min_pixels, image_max_pixels, vision_patch_size, vision_temporal_patch_size,
                            vision_spatial_merge_size),
        video_preprocessor_(video_min_pixels, video_max_pixels, vision_patch_size, vision_temporal_patch_size,
                            vision_spatial_merge_size),
        vision_temporal_patch_size_(vision_temporal_patch_size),
        vision_spatial_merge_size_(vision_spatial_merge_size),
        chat_preprocessor_(std::move(chat_template), renderLegacyQwen3_5ChatTemplate) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);
    chat_preprocessor_.setControlTokens(bpe_.controlTokens());
    // Qwen3.5 special tokens
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|object_ref_start|>");
    special_tokens_trie_.add(L"<|object_ref_end|>");
    special_tokens_trie_.add(L"<|box_start|>");
    special_tokens_trie_.add(L"<|box_end|>");
    special_tokens_trie_.add(L"<|quad_start|>");
    special_tokens_trie_.add(L"<|quad_end|>");
    special_tokens_trie_.add(L"<|vision_start|>");
    special_tokens_trie_.add(L"<|vision_end|>");
    special_tokens_trie_.add(L"<|vision_pad|>");
    special_tokens_trie_.add(L"<|image_pad|>");
    special_tokens_trie_.add(L"<|video_pad|>");
    special_tokens_trie_.add(L"<|fim_prefix|>");
    special_tokens_trie_.add(L"<|fim_middle|>");
    special_tokens_trie_.add(L"<|fim_suffix|>");
    special_tokens_trie_.add(L"<|fim_pad|>");
    special_tokens_trie_.add(L"<|repo_name|>");
    special_tokens_trie_.add(L"<|file_sep|>");
    special_tokens_trie_.add(L"<think>");
    special_tokens_trie_.add(L"</think>");
    special_tokens_trie_.add(L"<tool_call>");
    special_tokens_trie_.add(L"</tool_call>");
    special_tokens_trie_.add(L"<tool_response>");
    special_tokens_trie_.add(L"</tool_response>");
  }

  // Product constructor: geometry and chat-template backend come from the
  // model configuration; the template is discovered next to tokenizer.json
  // unless another model directory is given.
  Qwen3_5Tokenizer(const std::string& file_path, const Qwen3_5Config& config, std::filesystem::path model_directory = {})
      : Qwen3_5Tokenizer(file_path, config.image_min_pixels, config.image_max_pixels, config.vision_patch_size,
                         config.vision_temporal_patch_size, config.vision_spatial_merge_size, 4 * 32 * 32,
                         24 * 32 * 32 * 1024,
                         preprocessor::ChatPreprocessorConfig{
                             .backend = config.chat_template_backend,
                             .template_options = {.model_directory = model_directory.empty()
                                                                         ? std::filesystem::path(file_path).parent_path()
                                                                         : std::move(model_directory)}}) {}

  preprocessor::ChatTemplateBackend chatTemplateBackend() const noexcept { return chat_preprocessor_.backend(); }

  // The checkpoint's control tokens; see BPE::controlTokens().
  const std::vector<std::string>& controlTokens() const { return bpe_.controlTokens(); }

  // Renders the prompt text for one runner turn without media expansion.
  std::string renderChatTemplate(const Qwen3_5Message& message) const {
    return chat_preprocessor_.render(
        makeQwen3_5ChatTemplateRequest(message.prompt, message.image_paths.size(), !message.video_frames_thwc.isNil()));
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    qwen3_5Regex(str, splitted);
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

  std::string detokenizeBytes(int64_t pos_idx) {
    auto str = _detokenize(pos_idx);
    std::string bytes;
    bytes.reserve(str.size());
    for (wchar_t c : str) {
      const auto it = bytes_2_unicode_dict_inverse_.find(c);
      if (it == bytes_2_unicode_dict_inverse_.end()) {
        throw std::runtime_error("Qwen3.5 tokenizer encountered an unknown byte-unicode symbol");
      }
      bytes.push_back(static_cast<char>(it->second));
    }
    return bytes;
  }

  std::wstring detokenize(int64_t pos_idx) override {
    return {mllm::preprocessor::utf8string2WideString(detokenizeBytes(pos_idx))};
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("qwen3_5-tokenizer-i0")
                     .alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  ARGenerationOutputPast convertMessage(const Qwen3_5Message& message) {
    if (message.prompt.empty()) { throw std::invalid_argument("Qwen3.5 prompt must not be empty"); }
    static constexpr std::string_view kReservedMarkers[] = {
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    };
    for (const auto marker : kReservedMarkers) {
      if (message.prompt.find(marker) != std::string::npos) {
        throw std::invalid_argument("Qwen3.5 prompt must not inject reserved multimodal markers");
      }
    }

    const bool has_image = !message.image_paths.empty();
    const bool has_video = !message.video_frames_thwc.isNil();
    if (has_image && has_video) {
      throw std::invalid_argument("Qwen3.5 initial video input does not mix still images and video");
    }
    if (has_video && message.video_frame_indices.size() < 4) {
      throw std::invalid_argument("Qwen3.5 bounded video input requires at least four sampled frames");
    }
    if (!has_video && (!message.video_frame_indices.empty() || message.video_frames_per_second != 0.0)) {
      throw std::invalid_argument("Qwen3.5 video metadata requires video frames");
    }
    // Stage 1: chat template (legacy or official Jinja) renders the prompt with
    // one placeholder per media block. Stage 2 expands those placeholders the
    // way the official processor does, then the tokenizer stage expands per
    // patch tokens from the image/video geometry.
    auto applied_string = renderChatTemplate(message);
    if (has_image) {
      const auto placeholder_count = countOccurrences(applied_string, kQwen3_5ImagePlaceholder);
      if (placeholder_count != message.image_paths.size()) {
        throw std::runtime_error("Qwen3.5 chat template rendered " + std::to_string(placeholder_count)
                                 + " image placeholders for " + std::to_string(message.image_paths.size()) + " images");
      }
    } else if (has_video) {
      const auto placeholder = applied_string.find(kQwen3_5VideoPlaceholder);
      if (placeholder == std::string::npos || countOccurrences(applied_string, kQwen3_5VideoPlaceholder) != 1) {
        throw std::runtime_error("Qwen3.5 chat template must render exactly one video placeholder");
      }
      const auto timestamps = calculateQwen3_5VideoTimestamps(message.video_frame_indices, message.video_frames_per_second,
                                                              vision_temporal_patch_size_);
      applied_string.replace(placeholder, sizeof(kQwen3_5VideoPlaceholder) - 1, makeQwen3_5VideoMarkers(timestamps));
    }

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor pixel_values = Tensor::nil();
    Tensor image_grid_thw = Tensor::nil();
    Tensor pixel_values_videos = Tensor::nil();
    Tensor video_grid_thw = Tensor::nil();
    std::vector<int32_t> token_types;
    if (has_image) {
      std::tie(pixel_values, image_grid_thw) = image_preprocessor_(message.image_paths);
      const auto* grid = image_grid_thw.ptr<int32_t>();
      std::vector<int32_t> image_token_counts(message.image_paths.size());
      for (size_t image_index = 0; image_index < message.image_paths.size(); ++image_index) {
        const auto* image_grid = grid + image_index * 3;
        image_token_counts[image_index] =
            image_grid[0] * image_grid[1] * image_grid[2] / (vision_spatial_merge_size_ * vision_spatial_merge_size_);
      }
      const int64_t image_token_id = bpe_._lookup_vocab(L"<|image_pad|>");
      std::tie(ids, token_types) = expandQwen3_5ImagePlaceholders(ids, image_token_id, image_token_counts);
    } else if (has_video) {
      if (message.video_frames_thwc.shape().size() != 4
          || message.video_frames_thwc.shape()[0] != static_cast<int32_t>(message.video_frame_indices.size())) {
        throw std::invalid_argument("Qwen3.5 video metadata must contain one source index per decoded frame");
      }
      auto resized_video_frames = video_preprocessor_.resizeFrames(message.video_frames_thwc);
      std::tie(pixel_values_videos, video_grid_thw) = video_preprocessor_.flattenNormalizedPatches(resized_video_frames);
      const auto* grid = video_grid_thw.ptr<int32_t>();
      const int32_t frame_token_count = grid[1] * grid[2] / (vision_spatial_merge_size_ * vision_spatial_merge_size_);
      std::vector<int32_t> video_token_counts(grid[0], frame_token_count);
      const int64_t video_token_id = bpe_._lookup_vocab(L"<|video_pad|>");
      std::tie(ids, token_types) = expandQwen3_5VideoPlaceholders(ids, video_token_id, video_token_counts);
    }

    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen3_5-tokenizer-i0")
                          .alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    if (!has_image && !has_video) { return {{"sequence", sequence}}; }

    auto mm_token_type_ids =
        Tensor::empty({1, static_cast<int32_t>(token_types.size())}, kInt32, kCPU).setMemType(kNormal).alloc();
    std::copy(token_types.begin(), token_types.end(), mm_token_type_ids.ptr<int32_t>());
    if (has_image) {
      return {
          {"sequence", sequence},
          {"pixel_values", pixel_values},
          {"image_grid_thw", image_grid_thw},
          {"mm_token_type_ids", mm_token_type_ids},
      };
    }
    return {
        {"sequence", sequence},
        {"pixel_values_videos", pixel_values_videos},
        {"video_grid_thw", video_grid_thw},
        {"mm_token_type_ids", mm_token_type_ids},
    };
  }

 private:
  static size_t countOccurrences(const std::string& text, std::string_view needle) {
    size_t count = 0;
    for (auto pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + needle.size())) { ++count; }
    return count;
  }

  Qwen3_5ImagePreprocessor image_preprocessor_;
  Qwen3_5VideoPreprocessor video_preprocessor_;
  int32_t vision_temporal_patch_size_ = 2;
  int32_t vision_spatial_merge_size_ = 2;
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
  preprocessor::ChatPreprocessor chat_preprocessor_;
};

}  // namespace mllm::models::qwen3_5
