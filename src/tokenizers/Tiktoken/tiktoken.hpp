//
// Created by xwk on 25-1-8.
//

#ifndef MLLM_TIKTOKEN_HPP
#define MLLM_TIKTOKEN_HPP

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tokenizers/Tokenizer.hpp"

namespace mllm {
// define a type rank_t which is an alias of unsigned int
using rank_t = unsigned int;

// a map from string to rank_t
// i.e. token -> token id
// to prevent copying, we use string_view
using view_merge_rank_t = std::unordered_map<std::string_view, rank_t>;

// a map from string to rank_t
// i.e. token -> token id
using merge_rank_t = std::unordered_map<std::string, rank_t>;

std::vector<rank_t> byte_pair_encode(const view_merge_rank_t &ranks, const std::string &piece);

merge_rank_t load_tiktoken_bpe(const std::string &filename);

std::string base64_encode(const std::string &input);

std::string base64_decode(const std::string &input);

template <typename Value>
std::unordered_map<std::string_view, Value> convert_keys_to_string_views(
    const std::unordered_map<std::string, Value> &input_map) {
    std::unordered_map<std::string_view, Value> result_map;
    for (const auto &pair : input_map) {
        result_map[pair.first] = pair.second;
    }
    return result_map;
}

class CoreBPE {
public:
    CoreBPE(const merge_rank_t &encoder,
            const std::unordered_map<std::string, rank_t> &special_tokens_encoder,
            const std::string &pattern);

    std::vector<rank_t> encode_ordinary_naive(const std::string &text);

    std::vector<rank_t> encode_native(const std::string &text, std::unordered_set<std::string> allowed_special_tokens);

    std::string decode(const std::vector<rank_t> &tokens);

    std::string decode_token(rank_t token);

private:
    merge_rank_t encoder;
    view_merge_rank_t view_encoder;
    std::unordered_map<std::string, rank_t> special_tokens_encoder;
    std::unordered_map<rank_t, std::string> decoder;
    std::unordered_map<rank_t, std::string> special_tokens_decoder;
    std::vector<std::string> pattern;
    std::vector<std::string> special_tokens_pattern;
};

class TiktokenTokenizer : public Tokenizer {
public:
    TiktokenTokenizer(const string &filename, unordered_map<std::string, rank_t> special_tokens_encoder, const string &pattern) :
        core(load_tiktoken_bpe(filename), special_tokens_encoder, pattern) {
        for (auto &pair : special_tokens_encoder) {
            special_tokens.insert(pair.first);
        }
    }
    TiktokenTokenizer(const string &filename) :
        core(load_tiktoken_bpe(filename), {}, "") {
    }

    TiktokenTokenizer(const merge_rank_t &mergeable_ranks, const std::unordered_map<std::string, rank_t> &special_tokens_encoder, const std::string &pattern) :
        core(mergeable_ranks, special_tokens_encoder, pattern) {
        for (auto &pair : special_tokens_encoder) {
            special_tokens.insert(pair.first);
        }
    }

    void setSpecialToken(const string &bos, const string &eos, const string &unk, const string &nl) override {
        this->bos_token = bos;
        this->eos_token = eos;
    }

    void tokenize(const string &text, vector<token_id_t> &tokens, bool bos) override;

    string detokenize(const vector<token_id_t> &tokens) override;

protected:
    CoreBPE core;
    std::unordered_set<std::string> special_tokens;

private:
    std::string bos_token;
    std::string eos_token;
};

} // namespace mllm

#endif // MLLM_TIKTOKEN_HPP
