//
// Created by lx on 23-10-7.
//

#ifndef MLLM_TOKENIZER_HPP
#define MLLM_TOKENIZER_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
namespace mllm {
typedef unsigned int token_id_t;
typedef std::string token_t;
typedef struct TokenT {
    token_id_t token_id;
    float score;
} Token;
class Tokenizer {
protected:
    const static token_id_t TokenBos = 1;
    const static token_id_t TokenEos = 2;
    const static token_id_t TokenNl = 13;
    const static token_id_t TokenUnk = 0;
    std::unordered_map<token_t, token_id_t> vocab_map_;
    std::vector<Token> id_token_;

public:
    virtual void tokenize(const std::string &text, std::vector<token_id_t> &tokens) = 0;
};

} // namespace mllm

#endif // MLLM_TOKENIZER_HPP
