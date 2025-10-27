/**
 * @file tokenization_ds_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-24
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <map>
#include <locale>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include "tokenizers/Tokenizer.hpp"

namespace mllm {
inline std::wint_t ord(const wchar_t *str) {
    return static_cast<std::wint_t>(*str);
}

inline wchar_t chr(std::wint_t value) {
    return static_cast<wchar_t>(value);
}

// some OS has no en_US.UTF-8 but has C.UTF-8.
inline void initLocal(const std::string &local_name = "en_US.UTF-8") {
    try {
        std::locale::global(std::locale(local_name));
    } catch (const std::exception &e) {
        std::locale::global(std::locale("C.UTF-8"));
    }
}

inline bool isLetter(wchar_t c) {
    return std::iswalpha(c);
}

inline bool isDigit(wchar_t c) {
    return std::iswdigit(c);
}

std::string wideString2Utf8String(const std::wstring &wstr);

std::wstring utf8string2WideString(const std::string &str);

// same with gpt2.bytes_to_unicode
//
// same with qwen2.bytes_to_unicode
//
/*
Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to
whitespace/control characters the bpe code barfs on.

The reversible bpe codes work on unicode strings. This means you need a large # of unicode
characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
dataset you end up needing around 5K for decent coverage. This is a significant percentage of
your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
unicode strings.
*/
void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t> &dict);

std::string wideString2Utf8String(const std::wstring &wstr) {
    std::string result;
    for (wchar_t wc : wstr) {
        if (wc <= 0x7FU) {
            result.push_back(static_cast<char>(wc));
        } else if (wc <= 0x7FFU) {
            result.push_back(static_cast<char>(0xC0U | ((wc >> 6U) & 0x1FU)));
            result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
        } else if (wc <= 0xFFFFU) {
            result.push_back(static_cast<char>(0xE0U | ((wc >> 12U) & 0x0FU)));
            result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3FU)));
            result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
        } else if (wc <= 0x10FFFFU) {
            result.push_back(static_cast<char>(0xF0U | ((wc >> 18U) & 0x07U)));
            result.push_back(static_cast<char>(0x80U | ((wc >> 12U) & 0x3FU)));
            result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3FU)));
            result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
        }
    }
    return result;
}

std::wstring utf8string2WideString(const std::string &str) {
    std::wstring w_ret_string;
    for (unsigned int i = 0; i < str.size();) {
        auto byte = static_cast<unsigned char>(str[i]);
        if ((byte & 0x80U) == 0) {
            // 1-byte character
            w_ret_string.push_back(static_cast<wchar_t>(byte));
            ++i;
        } else if ((byte & 0xE0U) == 0xC0) {
            // 2-byte character
            if (i + 1 < str.size()) {
                wchar_t wc =
                    (static_cast<wchar_t>(byte & 0x1FU) << 6U) | (static_cast<wchar_t>(str[i + 1] & 0x3FU));
                w_ret_string.push_back(wc);
                i += 2;
            } else {
                break;
            }
        } else if ((byte & 0xF0U) == 0xE0U) {
            // 3-byte character
            if (i + 2 < str.size()) {
                wchar_t wc = (static_cast<wchar_t>(byte & 0x0FU) << 12U)
                             | (static_cast<wchar_t>(str[i + 1] & 0x3FU) << 6U)
                             | (static_cast<wchar_t>(str[i + 2] & 0x3FU));
                w_ret_string.push_back(wc);
                i += 3;
            } else {
                break;
            }
        } else if ((byte & 0xF8U) == 0xF0U) {
            // 4-byte character
            if (i + 3 < str.size()) {
                wchar_t wc = (static_cast<wchar_t>(byte & 0x07U) << 18U)
                             | (static_cast<wchar_t>(str[i + 1] & 0x3FU) << 12U)
                             | (static_cast<wchar_t>(str[i + 2] & 0x3FU) << 6U)
                             | (static_cast<wchar_t>(str[i + 3] & 0x3FU));
                w_ret_string.push_back(wc);
                i += 4;
            } else {
                break;
            }
        } else {
            // Invalid UTF-8 sequence
            ++i;
        }
    }
    return w_ret_string;
}

void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t> &dict) {
    std::vector<std::wint_t> bs((ord(L"~") - ord(L"!") + 1) + (ord(L"¬") - ord(L"¡") + 1)
                                + (ord(L"ÿ") - ord(L"®") + 1));

    int cnt = 0;
    for (std::wint_t i = ord(L"!"); i <= ord(L"~"); ++i) { bs[cnt++] = i; }
    for (std::wint_t i = ord(L"¡"); i <= ord(L"¬"); ++i) { bs[cnt++] = i; }
    for (std::wint_t i = ord(L"®"); i <= ord(L"ÿ"); ++i) { bs[cnt++] = i; }

    std::vector<std::wint_t> cs(bs.size());
    for (int i = 0; i < bs.size(); ++i) { cs[i] = bs[i]; }

    int n = 0;
    for (std::wint_t b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.emplace_back(b);
            cs.emplace_back(256 + n);
            ++n;
        }
    }

    std::vector<wchar_t> cs_chars(cs.size());
    for (int i = 0; i < cs.size(); ++i) { cs_chars[i] = chr(cs[i]); }
    for (int i = 0; i < bs.size(); ++i) { dict.insert({bs[i], cs_chars[i]}); }
}

// split text to tokens.
// > Trie.addSpecial("<|im_start|>")
// > Trie.split("<|im_start|>Hello world!")
//
// will give: ["<|im_start|>","Hello world!"]
class Trie {
    struct TrieNode {
        std::unordered_map<wchar_t, std::unique_ptr<TrieNode>> children;
        bool is_end = false;
    };

public:
    void add(const std::wstring &word);

    void update(const std::vector<std::wstring> &words);

    // I use FSA to implement the split function.
    std::vector<std::wstring> split(const std::wstring &text);

    bool isSpecialToken(const std::wstring &token);

private:
    std::unique_ptr<TrieNode> root_ = std::make_unique<TrieNode>();
    std::unordered_set<std::wstring> special_tokens_;
};

void Trie::add(const std::wstring &word) {
    if (word.empty()) return;
    special_tokens_.insert(word);

    TrieNode *current = root_.get();

    for (const auto &c : word) {
        if (!current->children.count(c)) { current->children[c] = std::make_unique<TrieNode>(); }
        current = current->children[c].get();
    }

    current->is_end = true;
}

void Trie::update(const std::vector<std::wstring> &words) {
    for (const auto &word : words) { add(word); }
}

std::vector<std::wstring> Trie::split(const std::wstring &text) {
    std::map<size_t, TrieNode *> states;
    std::vector<size_t> offsets = {0};
    size_t skip = 0;

    for (size_t current = 0; current < text.size(); ++current) {
        if (skip > current) continue;

        std::unordered_set<size_t> to_remove;
        bool reset = false;

        wchar_t current_char = text[current];

        for (auto &[_start, node] : states) {
            auto start = _start;
            if (node->is_end) {
                // trying to find the longest match
                size_t max_end = current;

                for (auto &[look_start, look_node] : states) {
                    if (look_start > start) break;

                    size_t lookahead = (look_start < start) ? current + 1 : current;
                    size_t end = lookahead;
                    TrieNode *ptr = look_node;

                    while (lookahead < text.size()) {
                        wchar_t ch = text[lookahead];

                        if (!ptr->children.count(ch)) break;

                        ptr = ptr->children[ch].get();
                        lookahead++;

                        if (ptr->is_end) {
                            start = look_start;
                            end = lookahead;
                            skip = lookahead;
                        }
                    }

                    if (ptr->is_end && end > max_end) { max_end = end; }
                }
                offsets.push_back(start);
                offsets.push_back(max_end);
                reset = true;
                break;
            }
            if (node->children.count(current_char)) {
                states[start] = node->children[current_char].get();
            } else {
                to_remove.insert(start);
            }
        }
        if (reset) {
            states.clear();
        } else {
            for (auto start : to_remove) { states.erase(start); }
        }
        if (current >= skip && root_->children.count(current_char)) {
            states[current] = root_->children[current_char].get();
        }
    }
    for (auto &[start, node] : states) {
        if (node->is_end) {
            offsets.push_back(start);
            offsets.push_back(text.size());
            break;
        }
    }

    std::sort(offsets.begin(), offsets.end());
    std::vector<std::wstring> result;
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i - 1] != offsets[i]) {
            result.push_back(text.substr(offsets[i - 1], offsets[i] - offsets[i - 1]));
        }
    }
    if (offsets[offsets.size() - 1] != text.size()) {
        result.push_back(text.substr(offsets[offsets.size() - 1]));
    }
    return result;
}

bool Trie::isSpecialToken(const std::wstring &token) {
    return special_tokens_.count(token);
}

struct BPEPairHash {
    std::size_t operator()(const std::pair<std::wstring, std::wstring> &key) const {
        std::size_t h1 = std::hash<std::wstring>{}(key.first + key.second);
        return h1;
    }
};

class BPE {
public:
    // BPE can accept sentence piece's json foramt.
    bool initFromSentencePieceJson(const std::string &vocab_file_path, const std::string &merge_file_path);

    std::vector<std::wstring> _bpe(const std::wstring &token);

    long _lookup_vocab(const std::wstring &token);

    std::wstring _lookup_inverse_vocab(long idx);

private:
    std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> _get_pairs(
        const std::vector<std::wstring> &word);

    std::unordered_map<std::wstring, long> vocab_;
    std::unordered_map<long, std::wstring> vocab_inverse_;
    std::unordered_map<std::pair<std::wstring, std::wstring>, long, BPEPairHash> bpe_ranks_;
};

bool BPE::initFromSentencePieceJson(const std::string &vocab_file_path, const std::string &merge_file_path) {
    // open vocab file.
    {
        std::ifstream file(vocab_file_path, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Read and verify magic number
        int32_t magic;
        if (!file.read(reinterpret_cast<char *>(&magic), sizeof(magic)) || magic != 23333) {
            return false;
        }

        int32_t vocab_size;
        if (!file.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size))) {
            return false;
        }

        for (int i = 0; i < vocab_size; ++i) {
            // Read token ID
            int32_t token_id;
            if (!file.read(reinterpret_cast<char *>(&token_id), sizeof(token_id))) {
                return false;
            }

            // Read token string
            int32_t str_len;
            if (!file.read(reinterpret_cast<char *>(&str_len), sizeof(str_len))) {
                return false;
            }

            std::string token(str_len, '\0');
            if (!file.read(token.data(), str_len)) {
                return false;
            }

            float score;
            if (!file.read(reinterpret_cast<char *>(&score), sizeof(score))) {
                return false;
            }

            auto str = utf8string2WideString(token);
            vocab_.insert({
                str,
                token_id,
            });
            vocab_inverse_.insert({
                token_id,
                str,
            });
        }
    }

    // open merge file.
    {
        std::ifstream file(merge_file_path);
        if (!file.is_open()) {
            return false;
        }

        std::string line;

        // there i sno need to jump a line.
        // std::getline(file, line);

        long cnt = 0;
        while (std::getline(file, line)) {
            std::string_view line_view(line);

            size_t space_pos = line_view.find(' ');
            if (space_pos == std::string_view::npos) {
                continue;
            }

            std::string first_str(line_view.substr(0, space_pos));
            std::string second_str(line_view.substr(space_pos + 1));

            std::wstring first = utf8string2WideString(first_str);
            std::wstring second = utf8string2WideString(second_str);

            bpe_ranks_.insert({{first, second}, cnt++});
        }
    }
    return true;
}

std::vector<std::wstring> BPE::_bpe(const std::wstring &token) {
    // TODO check cache

    std::vector<std::wstring> word;
    for (const auto &w : token) word.emplace_back(std::wstring{w});

    auto pairs = _get_pairs(word);
    if (pairs.empty()) return {token};

    while (true) {
        bool has_bigram = false;
        long rank_bigram = std::numeric_limits<long>::max();
        std::pair<std::wstring, std::wstring> bigram;

        for (const auto &p : pairs) {
            if (bpe_ranks_.count(p)) {
                auto rank = bpe_ranks_.at(p);
                if (rank < rank_bigram) {
                    rank_bigram = rank;
                    bigram = p;
                    has_bigram = true;
                }
            }
        }

        if (!has_bigram) { break; }

        auto [first, second] = bigram;
        std::vector<std::wstring> new_word;
        int i = 0;

        while (i < word.size()) {
            // Find the next occurrence of 'first' starting at i
            int j = i;
            while (j < word.size() && word[j] != first) { j++; }

            // Add elements from i to j-1 (if any)
            if (j > i) { new_word.insert(new_word.end(), word.begin() + i, word.begin() + j); }

            // Check if we can merge at position j
            if (j < word.size() - 1 && word[j] == first && word[j + 1] == second) {
                new_word.push_back(first + second);
                i = j + 2; // Skip both merged elements
            } else if (j < word.size()) {
                new_word.push_back(word[j]);
                i = j + 1;
            } else {
                i = j; // j == word.size()
            }
        }

        word = std::move(new_word);
        if (word.size() == 1) {
            break;
        } else {
            pairs = _get_pairs(word);
        }
    }

    return word;
}

long BPE::_lookup_vocab(const std::wstring &token) {
    if (vocab_.find(token) != vocab_.end()) {
        return vocab_[token];
    } else {
        return 0;
    }
}

std::wstring BPE::_lookup_inverse_vocab(long idx) {
    if (vocab_inverse_.find(idx) != vocab_inverse_.end()) {
        return vocab_inverse_[idx];
    } else {
        return L"";
    }
}

std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> BPE::_get_pairs(
    const std::vector<std::wstring> &word) {
    std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> pairs;
    if (word.size() < 2) return pairs;
    auto prev_char = word[0];
    for (size_t i = 1; i < word.size(); ++i) {
        pairs.insert({prev_char, word[i]});
        prev_char = word[i];
    }
    return pairs;
}

// we need to handle this:
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool deepSeekQwen2TokenizerMatchPattern(const std::wstring &str, size_t &pos,
                                        std::wstring &matched);

bool deepSeekQwen2Regex(const std::string &str, std::vector<std::wstring> &splitted);

bool deepSeekQwen2TokenizerMatchPattern(const std::wstring &str, size_t &pos,
                                        std::wstring &matched) {
    if (pos >= str.size()) return false;

    // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
    static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
    for (const auto &contraction : contractions) {
        if (pos + contraction.size() <= str.size()
            && str.compare(pos, contraction.size(), contraction) == 0) {
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
        if (!isLetter(str[pos]) && !isDigit(str[pos]) && str[pos] != L'\r'
            && str[pos] != L'\n') {
            matched += str[pos];
            ++pos;
            has_prefix = true;
        }

        // Require at least one letter
        if (pos < str.size() && isLetter(str[pos])) {
            do {
                matched += str[pos];
                ++pos;
            } while (pos < str.size() && isLetter(str[pos]));
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
    if (isDigit(str[pos])) {
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
        if (pos < str.size() && !std::iswspace(str[pos]) && !isLetter(str[pos])
            && !isDigit(str[pos])) {
            do {
                ++pos;
            } while (pos < str.size() && !std::iswspace(str[pos]) && !isLetter(str[pos])
                     && !isDigit(str[pos]));

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

bool deepSeekQwen2Regex(const std::string &str, std::vector<std::wstring> &splitted) {
    auto w_string = utf8string2WideString(str);
    size_t pos = 0;
    while (pos < w_string.size()) {
        std::wstring matched;
        if (deepSeekQwen2TokenizerMatchPattern(w_string, pos, matched)) {
            splitted.push_back(matched);
        } else {
            ++pos;
        }
    }
    return true;
}

class DeepSeekQWen2Tokenizer final : public Tokenizer {
public:
    explicit DeepSeekQWen2Tokenizer(const std::string &vocab_file, const std::string &merge_file, bool split_special_tokens = false) {
        initLocal();
        makeBytes2UnicodeMap(bytes_2_unicode_dict_);
        for (auto &kv : bytes_2_unicode_dict_) {
            bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first});
        }
        bpe_.initFromSentencePieceJson(vocab_file, merge_file);
        special_tokens_trie_.add(L"<｜end▁of▁sentence｜>");
        special_tokens_trie_.add(L"<｜begin▁of▁sentence｜>");
        special_tokens_trie_.add(L"<|quad_start|>");
        special_tokens_trie_.add(L"<|quad_end|>");
        special_tokens_trie_.add(L"<|vision_start|>");
        special_tokens_trie_.add(L"<|vision_end|>");
        special_tokens_trie_.add(L"<|vision_pad|>");
        special_tokens_trie_.add(L"<|image_pad|>");
        special_tokens_trie_.add(L"<|video_pad|>");
        special_tokens_trie_.add(L"<｜User｜>");
        special_tokens_trie_.add(L"<｜Assistant｜>");
    }

    void tokenize(const std::string &str, std::vector<token_id_t> &ret_token_ids, bool bos) override {
        auto tokens = special_tokens_trie_.split(utf8string2WideString(str));
        std::vector<std::wstring> all_tokens;
        for (const auto &token : tokens) {
            if (special_tokens_trie_.isSpecialToken(token)) {
                all_tokens.emplace_back(token);
                continue;
            }
            auto tmp_tokens = _tokenize(wideString2Utf8String(token));
            all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
        }

        std::vector<long> ids;
        ids.reserve(all_tokens.size() + 1);
        ids.emplace_back(bpe_._lookup_vocab(L"<｜begin▁of▁sentence｜>"));
        for (const auto &str : all_tokens) { ids.emplace_back(bpe_._lookup_vocab(str)); }

        for (auto &item : ids) { ret_token_ids.emplace_back(item); }
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        std::vector<token_id_t> ids;
        tokenize(text, ids, true);
        return Tokenizer::tokens2Input(ids);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        std::wstring ret;
        for (auto &pos_idx : tokens) {
            auto str = bpe_._lookup_inverse_vocab(pos_idx);
            std::string utf_8_str;
            for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
            ret += utf8string2WideString(utf_8_str);
        }
        return wideString2Utf8String(ret);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        std::vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {detokenize({token_idx}), token_idx};
    }

    std::pair<bool, std::string> postprocess(std::string &text) override {
        if (text != "<｜end▁of▁sentence｜>") return {true, text};
        return {false, ""};
    }

    std::vector<std::wstring> _tokenize(const std::string &str) {
        std::vector<std::wstring> ret;
        std::vector<std::wstring> splitted;
        deepSeekQwen2Regex(str, splitted);
        for (const auto &s : splitted) {
            auto utf_8_str = wideString2Utf8String(s);
            std::wstring mapped_str;
            for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

            auto bpe_ts = bpe_._bpe(mapped_str);

            for (const auto &bpe_t : bpe_ts) { ret.push_back(bpe_t); }
        }

        return ret;
    }

    std::string apply_chat_template(const std::string &text) override {
        return "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>" + text + "<｜Assistant｜>";
    }

private:
    Trie special_tokens_trie_;
    BPE bpe_;
    std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
    std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};
} // namespace mllm
