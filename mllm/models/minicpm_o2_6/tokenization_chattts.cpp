// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "tokenization_chattts.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::models::chattts {

ChatTTSTokenizer::ChatTTSTokenizer(const std::string& tokenizer_file, bool do_lower_case, const std::string& unk_token,
                                   const std::string& sep_token, const std::string& pad_token, const std::string& cls_token,
                                   const std::string& mask_token, bool tokenize_chinese_chars, bool strip_accents)
    : cls_token_(cls_token),
      sep_token_(sep_token),
      pad_token_(pad_token),
      mask_token_(mask_token),
      do_lower_case_(do_lower_case) {
  wordpiece_ = std::make_unique<mllm::preprocessor::WordPiece>();

  // Try to load from tokenizer.json first
  if (!wordpiece_->initFromTokenizerJson(tokenizer_file)) {
    // Fallback to vocab file
    if (!wordpiece_->initFromVocabFile(tokenizer_file, do_lower_case, strip_accents, unk_token)) {
      throw std::runtime_error("Failed to initialize WordPiece tokenizer");
    }
  }

  // Get special token IDs
  cls_token_id_ = wordpiece_->token2id(cls_token_);
  sep_token_id_ = wordpiece_->token2id(sep_token_);
  pad_token_id_ = wordpiece_->token2id(pad_token_);
  mask_token_id_ = wordpiece_->token2id(mask_token_);
}

std::vector<int64_t> ChatTTSTokenizer::encode(const std::string& str) {
  auto tokens = wordpiece_->tokenize(str);
  return wordpiece_->convert_tokens_to_ids(tokens);
}

std::string ChatTTSTokenizer::decode(const std::vector<int64_t>& ids) {
  std::string result;

  for (int64_t id : ids) {
    std::string token = wordpiece_->id2token(id);

    // Skip special tokens
    if (token == cls_token_ || token == sep_token_ || token == pad_token_) { continue; }

    // Remove WordPiece prefix ##
    if (token.size() >= 2 && token[0] == '#' && token[1] == '#') {
      result += token.substr(2);
    } else {
      // Add space before token (except for first token or after ##)
      if (!result.empty() && result.back() != ' ') { result += " "; }
      result += token;
    }
  }

  return result;
}

std::vector<std::string> ChatTTSTokenizer::tokenize(const std::string& str) { return wordpiece_->tokenize(str); }

std::string ChatTTSTokenizer::detokenize(const std::vector<std::string>& tokenized_str) {
  std::string result;

  for (const auto& token : tokenized_str) {
    // Skip special tokens
    if (token == cls_token_ || token == sep_token_ || token == pad_token_) { continue; }

    // Remove WordPiece prefix ##
    if (token.size() >= 2 && token[0] == '#' && token[1] == '#') {
      result += token.substr(2);
    } else {
      // Add space before token (except for first token)
      if (!result.empty() && result.back() != ' ') { result += " "; }
      result += token;
    }
  }

  return result;
}

std::vector<int64_t> ChatTTSTokenizer::build_inputs_with_special_tokens(const std::vector<int64_t>& token_ids_0) {
  std::vector<int64_t> output;
  output.reserve(token_ids_0.size() + 2);

  // [CLS] X [SEP]
  output.push_back(cls_token_id_);
  output.insert(output.end(), token_ids_0.begin(), token_ids_0.end());
  output.push_back(sep_token_id_);

  return output;
}

std::vector<int64_t> ChatTTSTokenizer::build_inputs_with_special_tokens(const std::vector<int64_t>& token_ids_0,
                                                                        const std::vector<int64_t>& token_ids_1) {
  std::vector<int64_t> output;
  output.reserve(token_ids_0.size() + token_ids_1.size() + 3);

  // [CLS] A [SEP] B [SEP]
  output.push_back(cls_token_id_);
  output.insert(output.end(), token_ids_0.begin(), token_ids_0.end());
  output.push_back(sep_token_id_);
  output.insert(output.end(), token_ids_1.begin(), token_ids_1.end());
  output.push_back(sep_token_id_);

  return output;
}

std::vector<int64_t> ChatTTSTokenizer::create_token_type_ids_from_sequences(const std::vector<int64_t>& token_ids_0,
                                                                            const std::vector<int64_t>& token_ids_1) {
  std::vector<int64_t> token_type_ids;

  if (token_ids_1.empty()) {
    // Single sequence: [CLS] + token_ids_0 + [SEP]
    size_t total_length = 1 + token_ids_0.size() + 1;
    token_type_ids.resize(total_length, 0);
  } else {
    // Sequence pair: [CLS] + token_ids_0 + [SEP] + token_ids_1 + [SEP]
    size_t first_length = 1 + token_ids_0.size() + 1;
    size_t second_length = token_ids_1.size() + 1;

    token_type_ids.resize(first_length, 0);
    token_type_ids.resize(first_length + second_length, 1);
  }

  return token_type_ids;
}

ARGenerationOutputPast ChatTTSTokenizer::convertText(const std::string& text) {
  // Tokenize text
  auto tokens = wordpiece_->tokenize(text);
  auto ids = wordpiece_->convert_tokens_to_ids(tokens);

  // Build inputs with special tokens
  auto input_ids = build_inputs_with_special_tokens(ids);

  // Create input tensor
  Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ static_cast<int32_t>(input_ids.size())}, kInt64, kCPU)
                        .setMemType(kNormal)
                        .setName("chattts-tokenizer-input")
                        .alloc();

  auto ptr = sequence.ptr<int64_t>();
  for (size_t i = 0; i < input_ids.size(); ++i) { ptr[i] = input_ids[i]; }

  // Create token type IDs
  auto token_type_ids_vec = create_token_type_ids_from_sequences(ids);
  Tensor token_type_ids = Tensor::empty({/*batch*/ 1, /*seq*/ static_cast<int32_t>(token_type_ids_vec.size())}, kInt64, kCPU)
                              .setMemType(kNormal)
                              .setName("chattts-tokenizer-token-type-ids")
                              .alloc();

  auto type_ptr = token_type_ids.ptr<int64_t>();
  for (size_t i = 0; i < token_type_ids_vec.size(); ++i) { type_ptr[i] = token_type_ids_vec[i]; }

  return {
      {"input_ids", sequence},
      {"token_type_ids", token_type_ids},
  };
}

}  // namespace mllm::models::chattts
