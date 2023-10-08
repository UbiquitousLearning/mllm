//
// Created by lx on 23-10-7.
//

#include "Bpe.hpp"
static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
void mllm::BPE::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool unk_byte_token = false) {
    size_t offset = 0;
    int idx = 0;
    while (offset < text.size()) {
        CharSymbol symbol;
        symbol.ch = text.c_str() + offset;
        symbol.length = std::min(text.size() - offset, utf8_len(text[offset]));
        symbol.last = idx - 1;
        symbol.next = text.size() - offset - symbol.length > 0 ? idx + 1 : -1;
        offset += symbol.length;
        idx++;
        symbols_.emplace_back(symbol);
    }
    for (int i = 1; i < symbols_.size(); ++i) {
        //        std::cout<<symbols_[i].ch<<std::endl;
        // Always Keep the single symbol
        tryMergeSymbol(i - 1, i);
    }
    while (!queue_.empty()) {
        auto item = queue_.top();
        queue_.pop();
        auto first = symbols_[item.start];
        auto last = symbols_[item.end];
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
            auto result = this->vocab_map_.find(std::string(symbols_[i].ch, symbols_[i].length));
            if (result != this->vocab_map_.end()) {
                tokens.emplace_back(result->second);
            } else {
                if (!unk_byte_token) {
                    tokens.emplace_back(mllm::BPE::TokenUnk);
                } else {
                    for (int j = 0; j < (int)symbols_[i].length; ++j) {
                        token_id_t token_id = static_cast<uint8_t>(symbols_[i].ch[j]) + 3;
                        tokens.push_back(token_id);
                    }
                }
            }
        }
    }
}
void mllm::BPE::tryMergeSymbol(size_t start, size_t end) {
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
void mllm::BPE::tokenize(const std::string &text, std::vector<token_id_t> &tokens) {
    this->tokenize(std::move(text), tokens, false);
}
