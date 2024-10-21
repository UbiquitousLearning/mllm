//
// Created by Xiang Li on 2023/12/2.
//
#include "trie.hpp"
#include "tokenizers/Tokenizer.hpp"
#ifndef MLLM_UNIGRAM_HPP
#define MLLM_UNIGRAM_HPP
namespace mllm {
const static float K_UNK_PENALTY = 10.0;
class UnigramTokenizer : public Tokenizer {
    struct BestPath {
        uint64_t id = 0;
        float best_path_score = 0.0;
        int64_t starts_at = -1;
    };
    Trie<char> trie_;

public:
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) override;
    explicit UnigramTokenizer(const std::string &vocab_file);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback);
    std::string detokenize(const std::vector<token_id_t> &tokens) override;
};
} // namespace mllm

#endif // MLLM_UNIGRAM_HPP
