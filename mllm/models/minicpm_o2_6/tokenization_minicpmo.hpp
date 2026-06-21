// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <unordered_map>
#include <regex>

#include "mllm/core/DataTypes.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/audio/Audio.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/models/minicpm_o2_6/image_preprocessor_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/audio_preprocessor_minicpmo.hpp"

namespace mllm::models::minicpmo {

// same with qwen2
// 参考: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen2_vl/tokenization_qwen2_vl.py
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
inline bool miniCPMOTokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+ (non-letter/digit followed by letters)
  {
    size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();

    // Check optional non-letter/digit prefix (excluding \r\n)
    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }

    // Require at least one letter
    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    } else {
      // Rollback if no letters after prefix
      if (has_prefix) {
        pos = original_pos;
        matched.clear();
      }
    }
  }

  // 3. Match \p{N} (digits)
  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  // 4. Match ?[^\s\p{L}\p{N}]+[\r\n]* (punctuation/symbols with optional space prefix)
  {
    size_t original_pos = pos;
    matched.clear();
    size_t start = pos;

    // Optional space
    if (str[pos] == L' ') { ++pos; }

    // Require at least one non-letter/digit/whitespace
    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));

      // Capture from start (after optional space) to current pos
      matched = str.substr(start, pos - start);

      // Capture trailing newlines
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    } else {
      // Rollback if no symbols found
      pos = original_pos;
    }
  }

  // 5. Match \s*[\r\n]+ (newlines with leading whitespace)
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

  // 6. Match \s+(?!\S) (whitespace not followed by non-space)
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    // Check if at end or followed by whitespace
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
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

inline bool miniCPMORegex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (miniCPMOTokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

struct MiniCPMOMessage {
  std::string prompt;
  std::string img_file_path;
  std::string audio_file_path;
  // TODO: system prompt should be configurable according to different scenarios
  std::string system_prompt =
      "You are a helpful assistant. You can accept video, audio and text input and output voice and text.";

  [[nodiscard]] std::string buildChatMessage(bool has_img = false, bool has_audio = false, bool gen_audio = true) const {
    std::string result = "";
    // System message
    if (!system_prompt.empty()) { result += "<|im_start|>user\n" + system_prompt + "<|im_end|>\n"; }

    result += "<|im_start|>user\n";

    // Image placeholder
    if (!img_file_path.empty()) { result += "(<image>./</image>)"; }

    // Audio placeholder
    if (!audio_file_path.empty()) { result += "(<audio>./</audio>)"; }

    if (!prompt.empty()) {
      result += "\n" + prompt;
      result += "<|im_end|>\n";
    }

    // Assistant prompt start
    result += "<|im_start|>assistant\n";
    if (gen_audio) { result += "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"; }

    return result;
  }
};

struct MiniCPMOInput {
  std::string prompt;
  std::string img_file_path = "";
  std::string audio_file_path = "";
  std::vector<std::string> image_paths = {};
  std::vector<std::string> audio_paths = {};
};

class MiniCPMOTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit MiniCPMOTokenizer(const std::string& file_path, int32_t patch_size = 14)
      : image_preprocessor_(patch_size), audio_preprocessor_(16000, 80, 160) {  // Whisper standard: 16kHz, 80 mel bins, 160 hop
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
    special_tokens_trie_.add(L"<|vision_start|>");
    special_tokens_trie_.add(L"<|vision_end|>");
    special_tokens_trie_.add(L"<|vision_pad|>");
    special_tokens_trie_.add(L"<|image_pad|>");
    special_tokens_trie_.add(L"<|video_pad|>");
    special_tokens_trie_.add(L"<tool_call>");
    special_tokens_trie_.add(L"</tool_call>");

    // Add UNK token as special token
    special_tokens_trie_.add(L"<unk>");

    // Image tokens
    special_tokens_trie_.add(L"<image>");
    special_tokens_trie_.add(L"</image>");
    special_tokens_trie_.add(L"<ref>");
    special_tokens_trie_.add(L"</ref>");
    special_tokens_trie_.add(L"<box>");
    special_tokens_trie_.add(L"</box>");
    special_tokens_trie_.add(L"<quad>");
    special_tokens_trie_.add(L"</quad>");
    special_tokens_trie_.add(L"<point>");
    special_tokens_trie_.add(L"</point>");
    special_tokens_trie_.add(L"<slice>");
    special_tokens_trie_.add(L"</slice>");
    special_tokens_trie_.add(L"<image_id>");
    special_tokens_trie_.add(L"</image_id>");
    special_tokens_trie_.add(L"<unit>");
    special_tokens_trie_.add(L"</unit>");

    // Audio tokens
    special_tokens_trie_.add(L"<asr>");
    special_tokens_trie_.add(L"</asr>");
    special_tokens_trie_.add(L"<query>");
    special_tokens_trie_.add(L"</query>");
    special_tokens_trie_.add(L"<|audio_start|>");
    special_tokens_trie_.add(L"<|audio_end|>");
    special_tokens_trie_.add(L"<|spk_bos|>");
    special_tokens_trie_.add(L"<|spk|>");
    special_tokens_trie_.add(L"<|spk_eos|>");
    special_tokens_trie_.add(L"<|tts_bos|>");
    special_tokens_trie_.add(L"<|tts_eos|>");
    special_tokens_trie_.add(L"<|listen|>");
    special_tokens_trie_.add(L"<|speak|>");
    special_tokens_trie_.add(L"<|interrupt|>");
    special_tokens_trie_.add(L"<|vad_start|>");
    special_tokens_trie_.add(L"<|vad_end|>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::minicpmo::miniCPMORegex(str, splitted);
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
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("minicpmo-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return ret;
  }

  ARGenerationOutputPast convertMessage(const MiniCPMOMessage& message) {
    bool has_image = !message.img_file_path.empty();
    bool has_audio = !message.audio_file_path.empty();

    auto applied_string = message.buildChatMessage(has_image, has_audio);

    // ========== Phase 1: Process and collect all modality data ==========

    // Image processing results
    std::vector<Tensor> img_tensors;
    std::vector<std::pair<int, int>> original_sizes;
    std::vector<std::pair<int, int>> tgt_sizes;
    std::vector<int> grid;

    // Audio processing results
    Tensor audio_features = Tensor::nil();
    int audio_length = 0;

    // Process image if present
    if (has_image) {
      auto [tensors, orig_size, target_sizes, img_grid] = image_preprocessor_.process(message.img_file_path);
      img_tensors = std::move(tensors);
      original_sizes = std::move(orig_size);
      tgt_sizes = std::move(target_sizes);
      grid = std::move(img_grid);
    }

    // Process audio if present
    if (has_audio) {
      auto audio_data = mllm::audio::readWAV(message.audio_file_path, 16000);
      audio_length = static_cast<int>(audio_data.size());
      audio_features = audio_preprocessor_.processAudioData(audio_data.data(), audio_length);
    }

    // ========== Phase 2: Replace placeholders in text ==========

    // Replace image placeholder with actual image tokens
    if (has_image) {
      std::regex img_pattern(R"(\(<image>\./</image>\))");
      std::vector<std::string> image_tags;
      std::sregex_iterator iter(applied_string.begin(), applied_string.end(), img_pattern);
      std::sregex_iterator end;

      for (; iter != end; ++iter) { image_tags.push_back(iter->str()); }

      std::vector<std::string> text_chunks;
      int32_t pos = 0;
      for (const auto& tag : image_tags) {
        auto found = applied_string.find(tag, pos);
        if (found != std::string::npos) {
          text_chunks.push_back(applied_string.substr(pos, found - pos));
          pos = found + tag.size();
        }
      }
      text_chunks.push_back(applied_string.substr(pos));

      std::string final_text = "";
      for (size_t i = 0; i < image_tags.size(); ++i) {
        final_text += text_chunks[i];
        final_text += image_preprocessor_.get_slice_image_placeholder(original_sizes[i], grid, i);
      }
      final_text += "\n";
      final_text += text_chunks.back();
      applied_string = final_text;
    }

    // Replace audio placeholder with actual audio tokens
    if (has_audio) {
      std::string audio_placeholder = audio_preprocessor_.getAudioPlaceholder(audio_length, false);
      size_t audio_placeholder_pos = applied_string.find("(<audio>./</audio>)");
      if (audio_placeholder_pos != std::string::npos) { applied_string.replace(audio_placeholder_pos, 19, audio_placeholder); }
    }

    // ========== Phase 3: Tokenize the final text ==========

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> input_ids_vec;
    input_ids_vec.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { input_ids_vec.emplace_back(bpe_._lookup_vocab(str)); }

    // ========== Phase 4: Calculate bounds for all modalities ==========

    std::vector<std::pair<int, int>> image_bounds;
    std::vector<std::pair<int, int>> audio_bounds;

    // Calculate image bounds if image is present
    if (has_image) {
      auto [_, bounds] = image_preprocessor_.calc_bounds(input_ids_vec, bpe_);
      image_bounds = std::move(bounds);
    }

    // Calculate audio bounds if audio is present
    if (has_audio) {
      int64_t audio_start_id = bpe_._lookup_vocab(L"<|audio_start|>");
      int64_t audio_end_id = bpe_._lookup_vocab(L"<|audio_end|>");
      audio_bounds = audio_preprocessor_.calcAudioBounds(input_ids_vec, audio_start_id, audio_end_id);
    }

    // ========== Phase 5: Convert everything to tensors ==========

    return convertToTensors(input_ids_vec, img_tensors, tgt_sizes, image_bounds, audio_features, audio_bounds);
  }

 private:
  ARGenerationOutputPast convertToTensors(std::vector<int64_t>& input_ids_vec, std::vector<Tensor>& img_tensors,
                                          std::vector<std::pair<int, int>>& tgt_sizes,
                                          std::vector<std::pair<int, int>>& image_bounds, Tensor& audio_features,
                                          std::vector<std::pair<int, int>>& audio_bounds) {
    ARGenerationOutputPast result;

    // Convert input_ids to Tensor
    if (!input_ids_vec.empty()) {
      Tensor input_ids_tensor =
          Tensor::empty({1, (int32_t)input_ids_vec.size()}, kInt64, kCPU).setMemType(kExtraInput).setName("input_ids").alloc();
      auto input_ids_ptr = input_ids_tensor.ptr<int64_t>();
      for (size_t i = 0; i < input_ids_vec.size(); ++i) { input_ids_ptr[i] = input_ids_vec[i]; }
      result["input_ids"] = input_ids_tensor;
    }

    // Convert img_tensors to single Tensor with padding
    if (!img_tensors.empty()) {
      int channels = img_tensors[0].shape()[0];
      int patch_size = img_tensors[0].shape()[1];
      int HW_patch_size = img_tensors[0].shape()[2];
      for (const auto& img_tensor : img_tensors) {
        if (img_tensor.shape()[2] > HW_patch_size) { HW_patch_size = img_tensor.shape()[2]; }
      }

      Tensor pixel_values = Tensor::empty({(int)img_tensors.size(), channels, patch_size, HW_patch_size}, kFloat32, kCPU)
                                .setMemType(kExtraInput)
                                .setName("pixel_values")
                                .alloc();
      auto pixel_values_ptr = pixel_values.ptr<float_t>();

      // Zero-initialize the entire tensor first for padding
      const size_t total_size = img_tensors.size() * channels * patch_size * HW_patch_size * sizeof(float_t);
      std::memset(pixel_values_ptr, 0, total_size);

      // Copy data using memcpy for better performance
      for (int b = 0; b < (int)img_tensors.size(); b++) {
        const int src_hw = img_tensors[b].shape()[2];
        const auto src_ptr = img_tensors[b].ptr<float>();

        for (int c = 0; c < channels; c++) {
          for (int p = 0; p < patch_size; p++) {
            const int src_offset = c * patch_size * src_hw + p * src_hw;
            const int dst_offset =
                b * channels * patch_size * HW_patch_size + c * patch_size * HW_patch_size + p * HW_patch_size;
            std::memcpy(pixel_values_ptr + dst_offset, src_ptr + src_offset, src_hw * sizeof(float_t));
          }
        }
      }

      result["pixel_values"] = pixel_values;
    }

    // Convert tgt_sizes to Tensor
    if (!tgt_sizes.empty()) {
      Tensor tgt_sizes_tensor =
          Tensor::empty({(int32_t)tgt_sizes.size(), 2}, kInt32, kCPU).setMemType(kExtraInput).setName("tgt_sizes").alloc();
      auto tgt_sizes_ptr = tgt_sizes_tensor.ptr<int32_t>();
      for (size_t i = 0; i < tgt_sizes.size(); ++i) {
        tgt_sizes_ptr[i * 2] = tgt_sizes[i].first;
        tgt_sizes_ptr[i * 2 + 1] = tgt_sizes[i].second;
      }
      result["tgt_sizes"] = tgt_sizes_tensor;
    }

    // Convert image_bounds to Tensor
    if (!image_bounds.empty()) {
      Tensor image_bounds_tensor = Tensor::empty({(int32_t)image_bounds.size(), 2}, kInt32, kCPU)
                                       .setMemType(kExtraInput)
                                       .setName("image_bounds")
                                       .alloc();
      auto image_bounds_ptr = image_bounds_tensor.ptr<int32_t>();
      for (size_t i = 0; i < image_bounds.size(); ++i) {
        image_bounds_ptr[i * 2] = image_bounds[i].first;
        image_bounds_ptr[i * 2 + 1] = image_bounds[i].second;
      }
      result["image_bounds"] = image_bounds_tensor;
    }

    // Add audio features if present
    if (!audio_features.isNil()) { result["audio_features"] = audio_features; }

    // Convert audio_bounds to Tensor
    if (!audio_bounds.empty()) {
      Tensor audio_bounds_tensor = Tensor::empty({(int32_t)audio_bounds.size(), 2}, kInt32, kCPU)
                                       .setMemType(kExtraInput)
                                       .setName("audio_bounds")
                                       .alloc();
      auto audio_bounds_ptr = audio_bounds_tensor.ptr<int32_t>();
      for (size_t i = 0; i < audio_bounds.size(); ++i) {
        audio_bounds_ptr[i * 2] = audio_bounds[i].first;
        audio_bounds_ptr[i * 2 + 1] = audio_bounds[i].second;
      }
      result["audio_bounds"] = audio_bounds_tensor;
    }

    return result;
  }

 private:
  // For image only.
  MiniCPMOImageProcessor image_preprocessor_;

  // For audio only.
  MiniCPMOAudioProcessor audio_preprocessor_;

  // For text
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::minicpmo
