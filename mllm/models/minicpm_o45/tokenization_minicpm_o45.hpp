// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "mllm/core/DataTypes.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/minicpm_o2_6/audio_preprocessor_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/image_preprocessor_minicpmo.hpp"
#include "mllm/preprocessor/audio/Audio.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::minicpm_o45 {

// Same tokenizer splitting rules as Qwen2/Qwen3 family.
inline bool miniCPMO45TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
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
    }

    if (has_prefix) {
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
    }

    pos = original_pos;
  }

  {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    }
    pos = start;
  }

  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline bool miniCPMO45Regex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (miniCPMO45TokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

struct MiniCPMO45Message {
  std::string prompt;
  std::string img_file_path;
  std::string audio_file_path;
  std::string system_prompt =
      "You are a helpful assistant. You can accept video, audio and text input and output voice and text.";

  [[nodiscard]] std::string buildChatMessage(bool generate_audio = false) const {
    std::string result;
    if (!system_prompt.empty()) { result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"; }

    result += "<|im_start|>user\n";
    if (!img_file_path.empty()) { result += "(<image>./</image>)"; }
    if (!audio_file_path.empty()) { result += "(<audio>./</audio>)"; }

    if (!prompt.empty()) {
      if (!img_file_path.empty() || !audio_file_path.empty()) { result += "\n"; }
      result += prompt;
    }

    result += "<|im_end|>\n";
    result += "<|im_start|>assistant\n";

    if (generate_audio) { result += "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"; }
    return result;
  }
};

class MiniCPMO45Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit MiniCPMO45Tokenizer(const std::string& tokenizer_path, int32_t patch_size = 14, int32_t audio_pool_step = 5)
      : image_preprocessor_(patch_size),
        audio_preprocessor_(16000, 80, 160),
        audio_pool_step_(audio_pool_step) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }

    bpe_.initFromSentencePieceJson(tokenizer_path);

    const std::vector<std::wstring> special_tokens = {
        L"<unk>",
        L"<|endoftext|>",
        L"<|im_start|>",
        L"<|im_end|>",
        L"<|object_ref_start|>",
        L"<|object_ref_end|>",
        L"<|box_start|>",
        L"<|box_end|>",
        L"<|quad_start|>",
        L"<|quad_end|>",
        L"<|vision_start|>",
        L"<|vision_end|>",
        L"<|vision_pad|>",
        L"<|image_pad|>",
        L"<|video_pad|>",
        L"<tool_call>",
        L"</tool_call>",
        L"<tool_response>",
        L"</tool_response>",
        L"<think>",
        L"</think>",
        L"<image>",
        L"</image>",
        L"<ref>",
        L"</ref>",
        L"<box>",
        L"</box>",
        L"<quad>",
        L"</quad>",
        L"<point>",
        L"</point>",
        L"<slice>",
        L"</slice>",
        L"<image_id>",
        L"</image_id>",
        L"<unit>",
        L"</unit>",
        L"<answer>",
        L"</answer>",
        L"<perception>",
        L"</perception>",
        L"<|audio_start|>",
        L"<|audio|>",
        L"<|audio_end|>",
        L"<|spk_bos|>",
        L"<|spk|>",
        L"<|spk_eos|>",
        L"<|tts_bos|>",
        L"<|tts_eos|>",
        L"<|listen|>",
        L"<|speak|>",
        L"<|interrupt|>",
        L"<|vad_start|>",
        L"<|vad_end|>",
        L"<|chunk_eos|>",
        L"<|chunk_bos|>",
        L"<|chunk_tts_bos|>",
        L"<|chunk_tts_eos|>",
    };

    for (const auto& token : special_tokens) { addSpecialToken(token); }
    loadSpecialTokensFromTokenizerJson(tokenizer_path);
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::minicpm_o45::miniCPMO45Regex(str, splitted);
    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

      auto bpe_tokens = bpe_._bpe(mapped_str);
      for (const auto& bpe_token : bpe_tokens) { ret.push_back(bpe_token); }
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
    for (wchar_t c : str) { utf_8_str.push_back(static_cast<unsigned char>(bytes_2_unicode_dict_inverse_[c])); }
    return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor ret = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("minicpmo45-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  int64_t lookupTokenId(const std::wstring& token) { return bpe_._lookup_vocab(token); }

  ARGenerationOutputPast convertMessage(const MiniCPMO45Message& message, bool generate_audio_prompt = false) {
    bool has_image = !message.img_file_path.empty();
    bool has_audio = !message.audio_file_path.empty();

    auto applied_string = message.buildChatMessage(generate_audio_prompt);

    std::vector<Tensor> img_tensors;
    std::vector<std::pair<int, int>> original_sizes;
    std::vector<std::pair<int, int>> tgt_sizes;
    std::vector<int> grid;

    Tensor audio_features = Tensor::nil();
    int32_t audio_length = 0;

    if (has_image) {
      auto [tensors, orig_size, target_sizes, img_grid] = image_preprocessor_.process(message.img_file_path);
      img_tensors = std::move(tensors);
      original_sizes = std::move(orig_size);
      tgt_sizes = std::move(target_sizes);
      grid = std::move(img_grid);
    }

    if (has_audio) {
      auto audio_data = mllm::audio::readWAV(message.audio_file_path, 16000);
      audio_length = static_cast<int32_t>(audio_data.size());
      audio_features = audio_preprocessor_.processAudioData(audio_data.data(), audio_length);
    }

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
          pos = static_cast<int32_t>(found + tag.size());
        }
      }
      text_chunks.push_back(applied_string.substr(pos));

      std::string final_text;
      for (size_t i = 0; i < image_tags.size(); ++i) {
        final_text += text_chunks[i];
        final_text += image_preprocessor_.get_slice_image_placeholder(original_sizes[i], grid, static_cast<int32_t>(i));
      }
      final_text += text_chunks.back();
      applied_string = final_text;
    }

    if (has_audio) {
      auto audio_placeholder = getAudioPlaceholder(audio_length, false);
      size_t audio_placeholder_pos = applied_string.find("(<audio>./</audio>)");
      if (audio_placeholder_pos != std::string::npos) {
        applied_string.replace(audio_placeholder_pos, std::string("(<audio>./</audio>)").size(), audio_placeholder);
      }
    }

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> input_ids_vec;
    input_ids_vec.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { input_ids_vec.emplace_back(bpe_._lookup_vocab(str)); }

    std::vector<std::pair<int, int>> image_bounds;
    std::vector<std::pair<int, int>> audio_bounds;

    if (has_image) {
      auto [_, bounds] = image_preprocessor_.calc_bounds(input_ids_vec, bpe_);
      image_bounds = std::move(bounds);
    }

    if (has_audio) {
      int64_t audio_start_id = bpe_._lookup_vocab(L"<|audio_start|>");
      int64_t audio_end_id = bpe_._lookup_vocab(L"<|audio_end|>");
      audio_bounds = audio_preprocessor_.calcAudioBounds(input_ids_vec, audio_start_id, audio_end_id);
    }

    return convertToTensors(input_ids_vec, img_tensors, tgt_sizes, image_bounds, audio_features, audio_bounds);
  }

 private:
  void addSpecialToken(const std::wstring& token) {
    if (!token.empty()) { special_tokens_trie_.add(token); }
  }

  void loadSpecialTokensFromTokenizerJson(const std::string& tokenizer_path) {
    std::ifstream in(tokenizer_path);
    if (!in.is_open()) { return; }

    nlohmann::json json_data;
    try {
      json_data = nlohmann::json::parse(in);
    } catch (...) {
      return;
    }

    if (!json_data.contains("added_tokens") || !json_data["added_tokens"].is_array()) { return; }
    for (const auto& token_info : json_data["added_tokens"]) {
      if (!token_info.contains("content")) { continue; }
      addSpecialToken(preprocessor::utf8string2WideString(token_info["content"].get<std::string>()));
    }
  }

  [[nodiscard]] std::string getAudioPlaceholder(int32_t audio_length, bool chunk_input, float chunk_length = 1.0f) const {
    int32_t capped_audio_length = std::min(audio_length, max_audio_samples_);
    int32_t feature_lens = static_cast<int32_t>(std::ceil(static_cast<float>(capped_audio_length) / hop_length_));
    feature_lens = (feature_lens - 1) / 2 + 1;

    auto output_lens = (feature_lens - audio_pool_step_) / audio_pool_step_ + 1;
    output_lens = std::max(output_lens, 0);

    if (!chunk_input) {
      std::string audio_placeholder = "<|audio_start|>";
      for (int32_t i = 0; i < output_lens; ++i) { audio_placeholder += "<unk>"; }
      audio_placeholder += "<|audio_end|>";
      return audio_placeholder;
    }

    auto fbank_feat_in_chunk = static_cast<int32_t>(chunk_length * 100);
    auto cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) / 2 + 1;
    auto audio_embeds_in_chunk = (cnn_feat_in_chunk - audio_pool_step_) / audio_pool_step_ + 1;
    audio_embeds_in_chunk = std::max(audio_embeds_in_chunk, 1);

    auto num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) / audio_embeds_in_chunk;

    std::string placeholders;
    int32_t total_unk_len = 0;
    for (int32_t i = 0; i < num_audio_chunks; ++i) {
      auto unk_len = std::min(audio_embeds_in_chunk, output_lens - total_unk_len);
      placeholders += "<|audio_start|>";
      for (int32_t j = 0; j < unk_len; ++j) { placeholders += "<unk>"; }
      placeholders += "<|audio_end|>";
      total_unk_len += unk_len;
    }
    return placeholders;
  }

  ARGenerationOutputPast convertToTensors(const std::vector<int64_t>& input_ids_vec, std::vector<Tensor>& img_tensors,
                                          const std::vector<std::pair<int, int>>& tgt_sizes,
                                          const std::vector<std::pair<int, int>>& image_bounds, const Tensor& audio_features,
                                          const std::vector<std::pair<int, int>>& audio_bounds) {
    ARGenerationOutputPast result;

    if (!input_ids_vec.empty()) {
      auto input_ids_tensor = Tensor::empty({1, static_cast<int32_t>(input_ids_vec.size())}, kInt64, kCPU)
                                  .setMemType(kExtraInput)
                                  .setName("input_ids")
                                  .alloc();
      auto* input_ids_ptr = input_ids_tensor.ptr<int64_t>();
      for (size_t i = 0; i < input_ids_vec.size(); ++i) { input_ids_ptr[i] = input_ids_vec[i]; }
      result["input_ids"] = input_ids_tensor;
    }

    if (!img_tensors.empty()) {
      int32_t channels = img_tensors[0].shape()[0];
      int32_t patch_size = img_tensors[0].shape()[1];
      int32_t hw_patch_size = img_tensors[0].shape()[2];
      for (const auto& img_tensor : img_tensors) {
        if (img_tensor.shape()[2] > hw_patch_size) { hw_patch_size = img_tensor.shape()[2]; }
      }

      auto pixel_values = Tensor::empty({static_cast<int32_t>(img_tensors.size()), channels, patch_size, hw_patch_size}, kFloat32,
                                        kCPU)
                              .setMemType(kExtraInput)
                              .setName("pixel_values")
                              .alloc();
      auto* pixel_values_ptr = pixel_values.ptr<float_t>();
      std::memset(pixel_values_ptr, 0, static_cast<size_t>(img_tensors.size()) * channels * patch_size * hw_patch_size * sizeof(float_t));

      for (int32_t b = 0; b < static_cast<int32_t>(img_tensors.size()); ++b) {
        int32_t src_hw = img_tensors[b].shape()[2];
        const auto* src_ptr = img_tensors[b].ptr<float>();

        for (int32_t c = 0; c < channels; ++c) {
          for (int32_t p = 0; p < patch_size; ++p) {
            int32_t src_offset = c * patch_size * src_hw + p * src_hw;
            int32_t dst_offset = b * channels * patch_size * hw_patch_size + c * patch_size * hw_patch_size + p * hw_patch_size;
            std::memcpy(pixel_values_ptr + dst_offset, src_ptr + src_offset, src_hw * sizeof(float_t));
          }
        }
      }

      result["pixel_values"] = pixel_values;
    }

    if (!tgt_sizes.empty()) {
      auto tgt_sizes_tensor = Tensor::empty({static_cast<int32_t>(tgt_sizes.size()), 2}, kInt32, kCPU)
                                  .setMemType(kExtraInput)
                                  .setName("tgt_sizes")
                                  .alloc();
      auto* tgt_sizes_ptr = tgt_sizes_tensor.ptr<int32_t>();
      for (size_t i = 0; i < tgt_sizes.size(); ++i) {
        tgt_sizes_ptr[i * 2] = tgt_sizes[i].first;
        tgt_sizes_ptr[i * 2 + 1] = tgt_sizes[i].second;
      }
      result["tgt_sizes"] = tgt_sizes_tensor;
    }

    if (!image_bounds.empty()) {
      auto image_bounds_tensor = Tensor::empty({static_cast<int32_t>(image_bounds.size()), 2}, kInt32, kCPU)
                                     .setMemType(kExtraInput)
                                     .setName("image_bounds")
                                     .alloc();
      auto* image_bounds_ptr = image_bounds_tensor.ptr<int32_t>();
      for (size_t i = 0; i < image_bounds.size(); ++i) {
        image_bounds_ptr[i * 2] = image_bounds[i].first;
        image_bounds_ptr[i * 2 + 1] = image_bounds[i].second;
      }
      result["image_bounds"] = image_bounds_tensor;
    }

    if (!audio_features.isNil()) { result["audio_features"] = audio_features; }

    if (!audio_bounds.empty()) {
      auto audio_bounds_tensor = Tensor::empty({static_cast<int32_t>(audio_bounds.size()), 2}, kInt32, kCPU)
                                     .setMemType(kExtraInput)
                                     .setName("audio_bounds")
                                     .alloc();
      auto* audio_bounds_ptr = audio_bounds_tensor.ptr<int32_t>();
      for (size_t i = 0; i < audio_bounds.size(); ++i) {
        audio_bounds_ptr[i * 2] = audio_bounds[i].first;
        audio_bounds_ptr[i * 2 + 1] = audio_bounds[i].second;
      }
      result["audio_bounds"] = audio_bounds_tensor;
    }

    return result;
  }

 private:
  minicpmo::MiniCPMOImageProcessor image_preprocessor_;
  minicpmo::MiniCPMOAudioProcessor audio_preprocessor_;
  int32_t audio_pool_step_ = 5;
  int32_t hop_length_ = 160;
  int32_t max_audio_samples_ = 30 * 16000;

  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::minicpm_o45
