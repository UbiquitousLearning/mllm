//
// Created by lx on 23-10-7.
//

#ifndef MLLM_BPE_HPP
#define MLLM_BPE_HPP
#include <queue>
#include "tokenizers/Tokenizer.hpp"

namespace mllm {
class BPE final : public Tokenizer {
    struct TokenItem {
        struct Compare {
            bool operator()(const TokenItem &lhs, const TokenItem &rhs) const {
                return (lhs.score < rhs.score) || (lhs.score == rhs.score && lhs.start > rhs.start); // Score Biggest, if same , pick first.
            }
        };
        size_t start;
        size_t end;
        float score;
        size_t length;
    };
    struct CharSymbol {
        const char *ch;
        int length;
        size_t last;
        size_t next;
    };
    std::vector<CharSymbol> symbols_;
    std::priority_queue<TokenItem, std::vector<TokenItem>, TokenItem::Compare> queue_;
    void tryMergeSymbol(size_t start, size_t end);

public:
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens) override;
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool unk_byte_token);
};
} // namespace mllm

#endif // MLLM_BPE_HPP
