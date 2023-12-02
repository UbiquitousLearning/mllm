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
void UnigramTokenizer::setSpecialToken(const std::string &bos, const std::string &eos, const std::string &unk, const std::string &nl){
    if (!bos.empty()) {
        auto bos_token = this->vocab_map_.find(bos);
        if (bos_token != this->vocab_map_.end()) {
            TokenBos = bos_token->second;
        } else {
            std::cerr << "BOS token not found in vocab file." << std::endl;
        }
    }
    if (!eos.empty()) {
        auto eos_token = this->vocab_map_.find(eos);
        if (eos_token != this->vocab_map_.end()) {
            TokenEos = eos_token->second;
        } else {
            std::cerr << "EOS token not found in vocab file." << std::endl;
        }
    }
    if (!unk.empty()) {
        auto unk_token = this->vocab_map_.find(unk);
        if (unk_token != this->vocab_map_.end()) {
            TokenUnk = unk_token->second;
        } else {
            std::cerr << "UNK token not found in vocab file." << std::endl;
        }
    }
}
void UnigramTokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) {
    this->tokenize(std::move(text), tokens, bos, true);
}
