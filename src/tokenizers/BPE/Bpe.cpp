//
// Created by lx on 23-10-7.
//

#include "Bpe.hpp"


static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
static std::vector<std::pair<string, string>> get_pairs( vector<string> word) {
    std::vector<std::pair<string, string>> pairs;
    for (int i = 1; i < word.size(); ++i) {
        pairs.emplace_back(word[i-1],word[i] );
    }
    return pairs;
}
vector<std::string> mllm::BPETokenizer::bpe(const std::string& token) {
    // static std::unordered_map<std::string, std::string> cache;

    // // 检查 token 是否在缓存中
    // if (cache.find(token) != cache.end()) {
    //     return cache[token];
    // }

    vector<string> word_splits = {};
    for (int i = 0; i < token.size()-1; ++i) {
        word_splits.emplace_back(token.substr(i, 1));
    }
    word_splits.emplace_back(token.substr(token.size()-1, 1)+"</w>");

    auto pairs = get_pairs(word_splits);
    if (pairs.empty()) {
        return {token + "</w>"};
    }

    while (true) {
        // 找到排名最小的字符对
        auto bigram = *std::min_element(pairs.begin(), pairs.end(), [this](const auto& a, const auto& b) {
            auto string_a = std::string(a.first)+" " + std::string(a.second);
            auto string_b = std::string(b.first)+" " + std::string(b.second);
            auto rank_a = merge_rank.count(string_a) ? merge_rank.at(string_a) : INFINITY;
            auto rank_b = merge_rank.count(string_b) ? merge_rank.at(string_b) : INFINITY;
            return rank_a < rank_b;
        });
        auto bigram_string = std::string(bigram.first)+" " + std::string(bigram.second);
        if (merge_rank.find(bigram_string) == merge_rank.end()) {
            break;
        }

        string first = bigram.first, second = bigram.second;
        vector<std::string> new_word;
        int i = 0;
        int j=0;
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
            if (word_splits[i]==first && i<word_splits.size() -1&& word_splits[i+1]==second){
                new_word.emplace_back(first+second);
                i+=2;
            }else {
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
    if (bos) {
        tokens.emplace_back(mllm::BPETokenizer::TokenBos);
    }
    if (!merge_rank.empty()){
//        std::cout<<"merge_rank is not empty! Loading"<<std::endl;
        // split text with space
        vector<string> words = {};
        for (int i = 0; i < text.size(); ++i) {
            if (text[i]==' '){
                words.emplace_back(text.substr(offset,i-offset));
                offset=i+1;
            }
        }
        words.emplace_back(text.substr(offset,text.size()-offset));
        for (const auto& word:words){
            auto word_splits = bpe(word);
            for (const auto& word_split:word_splits){
                if (auto result = this->vocab_map_.find(word_split); result != this->vocab_map_.end()) {
                    auto token_idx =  result->second ;
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
        if (TokenEos>0) {
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
