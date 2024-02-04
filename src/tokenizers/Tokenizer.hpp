//
// Created by Xiang Li on 23-10-7.
//

#ifndef MLLM_TOKENIZER_HPP
#define MLLM_TOKENIZER_HPP
#include <string>
#include <vector>
#include <unordered_map>
#ifdef ANDROID_API
#include <android/asset_manager.h>
#endif
#include "Tensor.hpp"
#include <Module.hpp>

namespace mllm {
class Net;
const static int VocabMagicNumber = 23333;
typedef unsigned int token_id_t;
typedef std::string token_t;
typedef struct TokenT {
    token_id_t token_id;
    token_t token;
    float score;
} Token;

/**
 * @brief A Tokenizer is used to tokenize a string into a vector of numbers.
 * Currently, all the models use the same tokenizer, and when it is initialized, it will load the vocabulary file.
 * Then it use either `BPE` or `Unigram` to tokenize the string.
 * Some models may have a preprocessing step, like `clip` or `fuyu`. When using them, you need to pass the tokenizer to the processor.
 */
class Tokenizer {
protected:
    inline  static token_id_t TokenBos = 1;
    inline  static token_id_t TokenEos = 2;
    inline  static token_id_t TokenNl = 13;
    inline  static token_id_t TokenUnk = 0;
    float min_score_ = 0.0;
    std::unordered_map<token_t, token_id_t> vocab_map_;
    std::vector<Token> id_token_;
    std::string vocab_file_name_;
// #ifdef ANDROID_API
//     AAssetManager *asset_manager_;
// #endif

    bool load_vocab(const std::string &vocab_file);

public:
    explicit Tokenizer(const std::string &vocab_file);
    virtual ~Tokenizer() {}
    virtual void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) = 0;
    virtual std::string detokenize(const std::vector<token_id_t> &tokens);
    void setSpecialToken(const std::string &bos = "", const std::string &eos = "", const std::string &unk = "", const std::string &nl = "");
    static std::string replaceString(const std::string &str, char old_char, const std::string &new_char);
    static std::string unCapitalize(const std::string &str);
    bool getTokenId(const token_t &token, token_id_t &id);
    bool isAvailible() const {
        return !this->vocab_map_.empty();
    }
    unsigned int getVocabSize() const {
        return this->vocab_map_.size();
    }
    static void token2Tensor(Net *net, vector<token_id_t> tokens, shared_ptr<Tensor> input_tensor);
    static void tokens2Tensor(Net *net, vector<vector<token_id_t>> tokens, shared_ptr<Tensor> input_tensor);
    static Tensor tokens2Input( vector<token_id_t> tokens_id, string name= "input", BackendType type = MLLM_CPU) {
        Tensor tensor1(1, 1, tokens_id.size(), 1, Module::backends[type], true);
        tensor1.setName(name);
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int idx = 0; idx < tokens_id.size(); ++idx) {
            tensor1.setDataAt<float>(0, 0, idx, 0, tokens_id[idx]);
        }
        return tensor1;
    }
};

} // namespace mllm

#endif // MLLM_TOKENIZER_HPP
