//
// Created by Xiang Li on 23-10-7.
//

#ifndef MLLM_BPE_HPP
#define MLLM_BPE_HPP
#include <queue>
#include "tokenizers/Tokenizer.hpp"
#include <unordered_map>
namespace mllm {
class BPETokenizer : public Tokenizer {
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
        int last;
        int next;
    };
    std::unordered_map<string, unsigned> merge_rank;
    std::vector<CharSymbol> symbols_;
    std::priority_queue<TokenItem, std::vector<TokenItem>, TokenItem::Compare> queue_;
    void tryMergeSymbol(size_t start, size_t end);
    std::unordered_map<unsigned char, std::string> bytes_to_unicode_;

public:
    const std::unordered_map<string, unsigned> getVocabMap() {
        return vocab_map_;
    }
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) override;
    vector<std::string> bpe(const std::string &token, std::string end_symbol);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, std::vector<std::string> &special_tokens, bool byte_fallback = false);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback, std::string end_symbol);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, const std::vector<std::string> &special);
    void setMergeRank(const std::unordered_map<string, unsigned> &merge_rank);
    explicit BPETokenizer(const std::string &vocab_file);
};
} // namespace mllm

#endif // MLLM_BPE_HPP
