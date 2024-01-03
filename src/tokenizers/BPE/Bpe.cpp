//
// Created by lx on 23-10-7.
//

#include "Bpe.hpp"
static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
static std::vector<std::pair<string, string>> get_pairs( std::string& word) {
    std::vector<std::pair<string, string>> pairs;
    for (int i = 0; i < word.size() - 1; ++i) {
        pairs.emplace_back(string(1,word[i]), string(1,word[i + 1]));
    }
    return pairs;
}
// std::string mllm::BPETokenizer::bpe(const std::string& token) {
//     static std::unordered_map<std::string, std::string> cache;
//
//     // 检查 token 是否在缓存中
//     if (cache.find(token) != cache.end()) {
//         return cache[token];
//     }
//
//     std::string word = token.substr(0, token.size() - 1) + token.back() + "</w>";
//     auto pairs = get_pairs(word);
//
//     if (pairs.empty()) {
//         return token + "</w>";
//     }
//
//     while (true) {
//         // 找到排名最小的字符对
//         auto bigram = *std::min_element(pairs.begin(), pairs.end(), [this](const auto& a, const auto& b) {
//             return merge_rank.count(a) ? merge_rank[a] : std::numeric_limits<int>::max();
//         });
//
//         auto merge_text = new char[200];
//             sprintf(merge_text,"%s %s", bigram.first, bigram.second);
//         if (merge_rank.find(bigram) == merge_rank.end()) {
//             break;
//         }
//
//         char first = bigram.first, second = bigram.second;
//         std::string new_word;
//         int i = 0;
//         while (i < word.size()) {
//             int j = word.find(first, i);
//             if (j == std::string::npos) {
//                 new_word += word.substr(i);
//                 break;
//             } else {
//                 new_word += word.substr(i, j - i);
//                 i = j;
//             }
//
//             if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
//                 new_word += first + second;
//                 i += 2;
//             } else {
//                 new_word += word[i];
//                 i += 1;
//             }
//         }
//         word = new_word;
//         if (word.size() == 1) {
//             break;
//         } else {
//             pairs = get_pairs(word);
//         }
//     }
//
//     cache[token] = word;
//     return word;
// }
void mllm::BPETokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback = false) {
    if (text.empty()) {
        return;
    }
    if (this->vocab_map_.empty() || this->id_token_.empty()) {
        std::cout << "The vocab map is empty!" << std::endl;
        return;
    }
    symbols_.clear();
    while (!queue_.empty()) queue_.pop();
    size_t offset = 0;
    int idx = 0;
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
        //        std::cout<<symbols_[i].ch<<std::endl;
        // Always Keep the single symbol
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
    if (bos) {
        tokens.emplace_back(mllm::BPETokenizer::TokenBos);
    }
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
mllm::BPETokenizer::BPETokenizer(const std::string &vocab_file) :
    Tokenizer(vocab_file) {
}
