//
// Created by 咸的鱼 on 2023/12/2.
//

#include "Unigram.hpp"
using namespace mllm;
UnigramTokenizer::UnigramTokenizer(const std::string &vocab_file):
    Tokenizer(std::move(vocab_file)) {

}
void UnigramTokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback) {
}

void UnigramTokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) {
    this->tokenize(std::move(text), tokens, bos, true);
}
