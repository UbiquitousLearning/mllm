//
// Created by 咸的鱼 on 2023/12/2.
//
#include "tokenizers/Tokenizer.hpp"
#ifndef MLLM_UNIGRAM_HPP
#define MLLM_UNIGRAM_HPP
namespace mllm {
class UnigramTokenizer final : public Tokenizer {
public:
void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) override;
    explicit UnigramTokenizer(const std::string &vocab_file);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback);
    void setSpecialToken(const std::string &bos="", const std::string &eos="", const std::string &unk="", const std::string &nl="");

};
} // namespace mllm

#endif // MLLM_UNIGRAM_HPP
