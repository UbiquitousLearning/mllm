//
// Created by lx on 23-10-7.
//

#ifndef MLLM_TOKENIZER_HPP
#define MLLM_TOKENIZER_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
namespace mllm {
const static int VocabMagicNumber = 23333;
typedef unsigned int token_id_t;
typedef std::string token_t;
typedef struct TokenT {
    token_id_t token_id;
    token_t token;
    float score;
} Token;
class Tokenizer {
protected:
    inline const static token_id_t TokenBos = 1;
    inline const static token_id_t TokenEos = 2;
    inline const static token_id_t TokenNl = 13;
    inline const static token_id_t TokenUnk = 0;
    std::unordered_map<token_t, token_id_t> vocab_map_;
    std::vector<Token> id_token_;
    bool load_vocab(const std::string &vocab_file);

public:
    virtual void tokenize(const std::string &text, std::vector<token_id_t> &tokens) = 0;
    Tokenizer(const std::string &vocab_file);
};

} // namespace mllm

#endif // MLLM_TOKENIZER_HPP
