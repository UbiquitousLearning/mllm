//
// Created by Xiang Li on 23-10-7.
//

#include "Bpe.hpp"
#include <iostream>
#include <regex>
#include <codecvt>
#include <unordered_map>

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
static std::vector<std::pair<string, string>> get_pairs(vector<string> word) {
    std::vector<std::pair<string, string>> pairs;
    for (int i = 1; i < word.size(); ++i) {
        pairs.emplace_back(word[i - 1], word[i]);
    }
    return pairs;
}
vector<std::string> mllm::BPETokenizer::bpe(const std::string &token, std::string end_symbol) {
    std::wstring_convert<std::codecvt_utf8_utf16<char32_t>, char32_t> converter;
    std::u32string u32_token = converter.from_bytes(token);

    std::vector<std::string> word_splits;
    for (char32_t c : u32_token) {
        word_splits.push_back(converter.to_bytes(c));
    }
    std::string last_str = word_splits.back();
    word_splits.pop_back();
    // word_splits.push_back(last_str + "</w>");
    word_splits.push_back(last_str + end_symbol);

    auto pairs = get_pairs(word_splits);
    if (pairs.empty()) {
        // return {token + "</w>"};
        return {token + end_symbol};
    }

    while (true) {
        auto bigram = *std::min_element(pairs.begin(), pairs.end(), [this](const auto &a, const auto &b) {
            auto string_a = std::string(a.first) + " " + std::string(a.second);
            auto string_b = std::string(b.first) + " " + std::string(b.second);
            auto rank_a = merge_rank.count(string_a) ? merge_rank.at(string_a) : INFINITY;
            auto rank_b = merge_rank.count(string_b) ? merge_rank.at(string_b) : INFINITY;
            return rank_a < rank_b;
        });
        auto bigram_string = std::string(bigram.first) + " " + std::string(bigram.second);
        if (merge_rank.find(bigram_string) == merge_rank.end()) {
            break;
        }

        string first = bigram.first, second = bigram.second;
        vector<std::string> new_word;
        int i = 0;
        int j = 0;
        while (i < word_splits.size()) {
            // find first at word_splits[i:]
            bool found = false;
            for (j = i; j < word_splits.size(); ++j) {
                if (word_splits[j] == first) {
                    found = true;
                    for (int k = i; k < j; ++k) {
                        new_word.emplace_back(word_splits[k]);
                    }
                    i = j;
                    break;
                }
            }
            if (!found) {
                new_word.insert(new_word.end(), word_splits.begin() + i, word_splits.end());
                break;
            }
            if (word_splits[i] == first && i < word_splits.size() - 1 && word_splits[i + 1] == second) {
                new_word.emplace_back(first + second);
                i += 2;
            } else {
                new_word.emplace_back(word_splits[i]);
                i++;
            }
        }
        word_splits = new_word;
        if (word_splits.size() == 1) {
            break;
        } else {
            pairs = get_pairs(word_splits);
        }
    }

    // cache[token] = word_splits;
    return word_splits;
}

void mllm::BPETokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, std::vector<std::string> &special_tokens, bool byte_fallback) {
    if (std::find(special_tokens.begin(), special_tokens.end(), text) != special_tokens.end()) {
        auto it = this->vocab_map_.find(text);
        if (it != this->vocab_map_.end()) {
            tokens.emplace_back(it->second);
            tokens.emplace_back(2.0);
        }
        return;
    }
    tokenize(text, tokens, bos, byte_fallback, "");
}

void mllm::BPETokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback = false, std::string end_symbol = "") {
    if (text.empty()) {
        return;
    }
    if (this->vocab_map_.empty() || this->id_token_.empty()) {
        MLLM_LOG_ERROR_STREAM << "The vocab map is empty!" << std::endl;
        return;
    }
    symbols_.clear();
    while (!queue_.empty()) queue_.pop();
    size_t offset = 0;
    int idx = 0;
    if (bos) {
        tokens.emplace_back(mllm::BPETokenizer::TokenBos);
    }
    if (!merge_rank.empty()) {
        std::vector<std::string> words;
        std::regex pattern("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|\\w+|\\d+|\\S+");
        std::smatch match;

        std::string::const_iterator searchStart(text.cbegin());
        while (std::regex_search(searchStart, text.cend(), match, pattern)) {
            words.push_back(match.str());
            searchStart = match.suffix().first;
        }

        for (const auto &word : words) {
            auto word_splits = bpe(word, end_symbol);
            for (const auto &word_split : word_splits) {
                if (auto result = this->vocab_map_.find(word_split); result != this->vocab_map_.end()) {
                    auto token_idx = result->second;
                    tokens.emplace_back(id_token_[token_idx].score);
                } else {
                    if (!byte_fallback) {
                        tokens.emplace_back(mllm::BPETokenizer::TokenUnk);
                    } else {
                        for (const char j : word_split) {
                            token_id_t token_id = static_cast<uint8_t>(j) + 3;
                            tokens.emplace_back(token_id);
                        }
                    }
                }
            }
        }
        if (TokenEos > 0) {
            tokens.push_back(TokenEos);
        }
        return;
    }
    while (offset < text.size()) {
        CharSymbol symbol;
        symbol.ch = text.c_str() + offset;
        symbol.length = std::min(text.size() - offset, utf8_len(text[offset]));
        symbol.last = idx - 1;
        symbol.next = text.size() - offset - symbol.length > 0 ? idx + 1 : -1;
        offset += symbol.length;
        symbols_.emplace_back(symbol);
        idx++;
    }
    for (int i = 1; i < symbols_.size(); ++i) {
        tryMergeSymbol(i - 1, i);
    }
    while (!queue_.empty()) {
        auto item = queue_.top();
        queue_.pop();
        auto &first = symbols_[item.start];
        auto &last = symbols_[item.end];
        if (first.length == 0 || last.length == 0) {
            continue;
        }
        // Maybe the symbol has been merged by other item
        if (first.length + last.length != item.length) {
            continue;
        }
        // Merge the symbol，make the first symbol as the merged symbol.
        first.length += last.length;
        last.length = 0;
        first.next = last.next;
        if (last.next != -1) {
            symbols_[last.next].last = item.start;
        }
        // Keep Merging!
        tryMergeSymbol(first.last, item.start);
        tryMergeSymbol(item.start, first.next);
    }
    // auto result = this->vocab_map_.find("<image>");
    // auto t = result->second;
    for (int i = 0; i < symbols_.size(); ++i) {
        if (symbols_[i].length > 0) {
            auto token_text = std::string(symbols_[i].ch, symbols_[i].length);
            auto result = this->vocab_map_.find(token_text);
            if (result != this->vocab_map_.end()) {
                tokens.emplace_back(result->second);
            } else {
                if (!byte_fallback) {
                    tokens.emplace_back(mllm::BPETokenizer::TokenUnk);
                } else {
                    for (int j = 0; j < (int)symbols_[i].length; ++j) {
                        token_id_t token_id = static_cast<uint8_t>(symbols_[i].ch[j]) + 3;
                        tokens.emplace_back(token_id);
                    }
                }
            }
        }
    }
}

void mllm::BPETokenizer::setMergeRank(const std::unordered_map<string, unsigned> &merge_rank) {
    this->merge_rank = merge_rank;
}

void mllm::BPETokenizer::tryMergeSymbol(size_t start, size_t end) {
    if (start == -1 || end == -1) {
        return;
    }
    std::string merge_str = std::string(symbols_[start].ch, symbols_[end].ch + symbols_[end].length);
    auto result = this->vocab_map_.find(merge_str);
    if (result != this->vocab_map_.end() && result->second < id_token_.size()) {
        auto token = this->id_token_[result->second];
        TokenItem item;
        item.start = start;
        item.end = end;
        item.score = token.score;
        item.length = merge_str.size();
        queue_.emplace(item);
    }
}
void mllm::BPETokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) {
    this->tokenize(std::move(text), tokens, bos, true);
}
std::pair<size_t, std::string> find_special(const std::string &text, const std::vector<std::string> &special, size_t pos) {
    for (const std::string &delimiter : special) {
        size_t found = text.find(delimiter, pos);
        if ((found != std::string::npos)) {
            return {found, delimiter};
        }
    }
    return {std::string::npos, ""};
}

void mllm::BPETokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, const std::vector<std::string> &special) {
    tokens.push_back(1);
    size_t startPos = 0;
    auto result = find_special(text, special, startPos);
    size_t found = result.first;
    std::string delimiter = result.second;
    while (found != std::string::npos) {
        vector<token_id_t> tokens_id = {};
        if (found > startPos) {
            this->tokenize(text.substr(startPos, found - startPos), tokens_id, true, true);
            tokens.insert(tokens.end(), tokens_id.begin() + 1, tokens_id.end() - 1);
        }
        std::string delimiter_;
        if (delimiter == "\n") {
            delimiter_ = "<0x0A>";
        } else {
            delimiter_ = delimiter;
        }
        auto result = this->vocab_map_.find(delimiter_);
        if (result != this->vocab_map_.end()) {
            tokens.push_back(result->second);
        }
        startPos = found + delimiter.length();
        auto result_ = find_special(text, special, startPos);
        found = result_.first;
        delimiter = result_.second;
    }
    if (startPos < text.length()) {
        vector<token_id_t> tokens_id = {};
        this->tokenize(text.substr(startPos), tokens_id, true, true);
        tokens.insert(tokens.end(), tokens_id.begin() + 1, tokens_id.end() - 1);
    }
}
/**
Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
characters the bpe code barfs on.

The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
tables between utf-8 bytes and unicode strings.
**/
static std::unordered_map<unsigned char, string> bytes_to_unicode() {
    std::unordered_map<unsigned char, string> mapping;
    unsigned char n = 0;

    for (int b = 0; b < 256; ++b) {
        int convert_uchar = 0;
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            convert_uchar = b;
        } else {
            convert_uchar = 256 + n;
            ++n;
        }

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        std::string s = converter.to_bytes(static_cast<char32_t>(convert_uchar));
        mapping[b] = s;
    }
    return mapping;
}

mllm::BPETokenizer::BPETokenizer(const std::string &vocab_file) :
    Tokenizer(vocab_file) {
    bytes_to_unicode_ = bytes_to_unicode();
}
