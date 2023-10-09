//
// Created by lx on 23-10-7.
//
#include "ParamLoader.hpp"
#include "Tokenizer.hpp"
/* Vocab Structure
 * ┌──────┬──────┬─────┬────────┬──────┬──────┬───────┐
 * │      │      │     │        │      │      │       │
 * │      │      │     │        │      │      │       │
 * │      │      │     │        │      │      │       │
 * │      │ List │Vocab│ Vocab  │ Token│ Score│       │
 * │ MAGIC│ Len  │ Len │        │  ID  │      │ ...   │
 * │      │ INT  │ INT │ String │  INT │ FP32 │       │
 * │      │      │     │        │      │      │       │
 * │      │      │     │        │      │      │       │
 * └──────┴──────┴─────┴────────┴──────┴──────┴───────┘
 */
namespace mllm {
bool Tokenizer::load_vocab(const std::string &vocab_file) {
    FILE *fp = fopen(vocab_file.c_str(), "r");
    if (fp == nullptr) {
        std::cout << "open file failed" << std::endl;
        return false;
    }
    fseek(fp, 0, SEEK_CUR);
    if (readInt(fp) != VocabMagicNumber) {
        std::cout << "magic number error" << std::endl;
        return false;
    }
    auto length = readInt(fp);
    if (length <= 0) {
        std::cout << "vocab length error" << std::endl;
        return false;
    }
    this->vocab_map_.reserve(length);
    this->id_token_.resize(length);
    int offset = 0;
    while (offset < length) {
        auto id = readInt(fp);
        auto token = readString(fp);
        auto score = readf32(fp);
        this->vocab_map_[token] = id;

        this->id_token_[id].score = score;
        this->id_token_[id].token_id = id;
        this->id_token_[id].token = token;
        offset++;
    }
    return true;
}
Tokenizer::Tokenizer(const std::string &vocab_file) {
    load_vocab(vocab_file);
}
string Tokenizer::detokenize(const vector<token_id_t> &tokens) {
    int size = tokens.size() - 1;
    string result;
    for (int i = 0; i < size; i++) {
        auto token_id = tokens[i];
        if (token_id == TokenUnk) {
            result += "<unk>";
            continue;
        }
        if (token_id == TokenBos) {
            continue;
        }
        if (token_id == TokenEos) {
            if (i != size - 1) {
                result += " ";
            }
        }
        result += this->id_token_[token_id].token;
    }
}
} // namespace mllm