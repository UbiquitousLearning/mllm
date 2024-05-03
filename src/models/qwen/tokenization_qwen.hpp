/**
 * @file tokenization_qwen.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-04-29
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_QWEN_HPP
#define TOKENIZATION_QWEN_HPP

// regex
#include <re2/re2.h>

#include "tokenizers/Tokenizer.hpp"
#include <algorithm>
#include <optional>
#include <unordered_map>

// unicode
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>

using namespace mllm;

namespace mllm {

static std::string any_to_utf8(std::string s) {
    icu::UnicodeString us(s.c_str());
    std::string utf8;
    us.toUTF8String(utf8);
    return utf8;
}

static auto _byte_pair_merge(
    const std::string &piece,
    const std::unordered_map<std::string, int> &ranks,
    std::function<int(int, int)> func) -> std::vector<int> {
    std::vector<std::pair<int, int>> parts;
    parts.reserve(piece.size() + 1);
    for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
        parts.emplace_back(idx, std::numeric_limits<int>::max());
    }

    auto get_rank = [&piece, &ranks](
                        const std::vector<std::pair<int, int>> &parts,
                        int start_idx,
                        int skip) -> std::optional<int> {
        if (start_idx + skip + 2 < parts.size()) {
            auto s = parts[start_idx].first;
            auto e = parts[start_idx + skip + 2].first;
            auto key = piece.substr(s, e - s);
            auto iter = ranks.find(key);
            if (iter != ranks.end()) {
                return iter->second;
            }
        }
        return std::nullopt;
    };

    for (auto i = 0U; i < parts.size() - 2; ++i) {
        auto rank = get_rank(parts, i, 0);
        if (rank) {
            assert(*rank != std::numeric_limits<int>::max());
            parts[i].second = *rank;
        }
    }

    while (true) {
        if (parts.size() == 1) break;

        auto min_rank = std::make_pair<int, int>(std::numeric_limits<int>::max(), 0);
        for (auto i = 0U; i < parts.size() - 1; ++i) {
            auto rank = parts[i].second;
            if (rank < min_rank.first) {
                min_rank = {rank, i};
            }
        }

        if (min_rank.first != std::numeric_limits<int>::max()) {
            auto i = min_rank.second;
            auto rank = get_rank(parts, i, 1);
            if (rank) {
                parts[i].second = *rank;
            } else {
                parts[i].second = std::numeric_limits<int>::max();
            }
            if (i > 0) {
                auto rank = get_rank(parts, i - 1, 1);
                if (rank) {
                    parts[i - 1].second = *rank;
                } else {
                    parts[i - 1].second = std::numeric_limits<int>::max();
                }
            }

            parts.erase(parts.begin() + (i + 1));
        } else {
            break;
        }
    }
    std::vector<int> out;
    out.reserve(parts.size() - 1);
    for (auto i = 0U; i < parts.size() - 1; ++i) {
        out.push_back(func(parts[i].first, parts[i + 1].first));
    }
    return out;
}

static auto byte_pair_encode(
    const std::string &piece,
    const std::unordered_map<std::string, int> &ranks) -> std::vector<int> {
    if (piece.size() == 1) {
        return {ranks.at(piece)};
    }

    auto func = [&piece, &ranks](int start, int stop) -> int {
        std::string key = piece.substr(start, stop - start);

        // key to utf-8
        auto nkey = any_to_utf8(key);

        return ranks.at("");
    };

    return _byte_pair_merge(piece, ranks, func);
}

class TikTokenizer final : public Tokenizer {
public:
    explicit TikTokenizer(const std::string &vocab_file, const std::string &pattern) :
        Tokenizer(vocab_file) {
        regex_ = std::make_unique<re2::RE2>("(" + pattern + ")");
    }

    auto _encode_ordinary_native(const std::string &text) const -> std::vector<int> {
        std::vector<int> ret;
        re2::StringPiece input(text);

        std::string piece;
        while (re2::RE2::FindAndConsume(&input, *regex_, &piece)) {
            auto iter = encoder_.find(piece);
            if (iter != encoder_.end()) {
                ret.push_back(iter->second);
                continue;
            }
            auto tokens = byte_pair_encode(piece, encoder_);
            ret.insert(ret.end(), tokens.begin(), tokens.end());
        }
        return ret;
    }

    auto encode_ordinary(const std::string &text) const -> std::vector<int> {
        return _encode_ordinary_native(text);
    }

    auto encode(const std::string &text) const -> std::vector<int> {
        return _encode_native(text, special_tokens_encoder).first;
    }

    auto _encode_native(
        const std::string &text,
        const std::unordered_map<std::string, int> &allowed_special) const -> std::pair<std::vector<int>, int> {
        std::vector<int> ret;
        int last_piece_token_len = 0;
        re2::StringPiece input(text);

        while (true) {
            auto [special, sub_input] = split_with_allowed_special_token(input, allowed_special);
            std::string piece;
            while (re2::RE2::FindAndConsume(&sub_input, *regex_, &piece)) {
                auto iter = encoder_.find(piece);
                if (iter != encoder_.end()) {
                    last_piece_token_len = 1;
                    ret.push_back(iter->second);
                    continue;
                }
                auto tokens = byte_pair_encode(piece, encoder_);
                last_piece_token_len = tokens.size();
                ret.insert(ret.end(), tokens.begin(), tokens.end());
            }

            if (special) {
                int token = special_tokens_encoder.at(*special);
                ret.push_back(token);
                last_piece_token_len = 0;
            } else {
                break;
            }
        }

        return {ret, last_piece_token_len};
    }

    auto split_with_allowed_special_token(
        re2::StringPiece &input,
        const std::unordered_map<std::string, int> &allowed_special) const -> std::pair<std::optional<std::string>, re2::StringPiece> {
        if (special_regex_ == nullptr) return {std::nullopt, input};

        auto start = input.begin();
        std::string special;
        while (true) {
            if (!re2::RE2::FindAndConsume(&input, *special_regex_, &special)) {
                break;
            }

            if (allowed_special.count(special) == 1) {
                return {std::move(special), re2::StringPiece(start, input.begin() - start - special.size())};
            }
        }

        return {std::nullopt, input};
    }

    auto decode(const std::vector<int> &tokens) const -> std::string {
        return _decode_native(tokens);
    }

    auto _decode_native(const std::vector<int> &tokens) const -> std::string {
        std::string ret;
        ret.reserve(tokens.size() * 2);
        for (auto token : tokens) {
            std::string token_bytes;
            auto iter = decoder_.find(token);
            if (iter != decoder_.end()) {
                token_bytes = iter->second;
            } else {
                iter = special_tokens_decoder.find(token);
                if (iter != special_tokens_decoder.end()) {
                    token_bytes = iter->second;
                } else {
                    throw std::runtime_error("unknown token: " + std::to_string(token));
                }
            }
            ret += token_bytes;
        }
        return ret;
    }

    void init(std::unordered_map<std::string, int> special_encoder = {}) {
        // init special encoder
        std::string special_pattern;
        for (const auto &item : special_encoder) {
            if (!special_pattern.empty()) {
                special_pattern += "|";
            }
            special_pattern += re2::RE2::QuoteMeta(item.first);
        }
        if (special_pattern.empty()) {
            special_regex_ = nullptr;
        } else {
            special_regex_ = std::make_unique<re2::RE2>("(" + special_pattern + ")");
        }

        // encoder to enxoder_
        for (auto item : vocab_map_) encoder_.emplace(item.first, item.second);
        for (const auto &[k, v] : encoder_) {
            decoder_.emplace(v, k);
        }
        assert(encoder_.size() == decoder_.size() && "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?");

        for (const auto &[k, v] : special_tokens_encoder) {
            special_tokens_decoder.emplace(v, k);
        }
    }

    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) override {
        auto _ = encode(text);
        for (auto it : _) tokens.emplace_back(it);
        // no need to add bos
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        std::vector<int> _;
        for (auto it : tokens) _.emplace_back(it);
        return decode(_);
    }

private:
    std::unique_ptr<re2::RE2> regex_;
    std::unique_ptr<re2::RE2> special_regex_;
    std::unordered_map<std::string, int> encoder_;
    std::unordered_map<std::string, int> special_tokens_encoder;
    std::unordered_map<int, std::string> decoder_;
    std::unordered_map<int, std::string> special_tokens_decoder;
};
} // namespace mllm

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

class QWenTokenizer final {
public:
    explicit QWenTokenizer(const std::string &vocab_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new TikTokenizer(vocab_file, PAT_STR);

        // TODO init special encoder;
        std::pair<std::string, int> bos_token;
        std::pair<std::string, int> eos_token;
        std::unordered_map<std::string, int> special_encoder_tmp = {
            // TODO
        };
        tokenizer->init(special_encoder_tmp);
        tokenizer->setSpecialToken(/*bos*/ bos_token.first, /*eos*/ eos_token.first);
    }

    ~QWenTokenizer() {
        delete tokenizer;
    }

    Tensor tokenize(std::string &text, int str_i = 0) {
        std::vector<token_id_t> ret;
        tokenizer->tokenize(text, ret, /*bos*/ false);
        return Tokenizer::tokens2Input(ret);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        // TODO argmax can't be used here due to utf-8.
    }

private:
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    TikTokenizer *tokenizer;
    std::unordered_map<char, std::string> byte_encoder;

public:
    token_id_t eos_id = 151643,
               bos_id = 151643, im_start_id = 151644, im_end_id = 151645;
};

#endif //! TOKENIZATION_QWEN_HPP