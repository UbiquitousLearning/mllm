//
// Created by Xiang Li on 23-10-7.
//
#include "ParamLoader.hpp"
#include "Tokenizer.hpp"
#include <Net.hpp>
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
    // #ifdef ANDROID_API
    //    auto *fp= AAssetManager_open(asset_manager_, vocab_file.c_str(), AASSET_MODE_RANDOM);
    // #else

    FILE *fp = fopen(vocab_file.c_str(), "rb");
    // #endif

    if (fp == nullptr) {
        std::cout << vocab_file << " open file failed" << std::endl;
        return false;
    }
    // Use a unique_ptr with a custom deleter to ensure the file is closed.

    std::unique_ptr<mllm_file, decltype(&fclose)> fp_guard(fp, &fclose);
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
    float min_score = INFINITY;
    this->vocab_map_.reserve(length);
    this->id_token_.resize(length);
    int offset = 0;
    while (offset < length) {
        auto id = readInt(fp);
        auto token = readString(fp);
        auto score = readf32(fp);
        this->vocab_map_[token] = id;
        if (score < min_score) {
            min_score = score;
        }
        this->id_token_[id].score = score;
        this->id_token_[id].token_id = id;
        this->id_token_[id].token = token;
        offset++;
    }
    this->min_score_ = min_score;
    return true;
}

std::string Tokenizer::replaceString(const std::string &str, char old_char, const std::string &new_char) {
    std::string result;
    for (auto &ch : str) {
        if (ch == old_char) {
            result += new_char;
        } else {
            result += ch;
        }
    }
    return result;
}

std::string Tokenizer::unCapitalize(const std::string &str) {
    std::string result;
    for (auto &ch : str) {
        if (ch >= 'A' && ch <= 'Z') {
            result += ch + 32;
        } else {
            result += ch;
        }
    }
    return result;
}

bool Tokenizer::getTokenId(const token_t &token, token_id_t &id) {
    auto token_id = this->vocab_map_.find(token);
    if (token_id != this->vocab_map_.end()) {
        id = token_id->second;
        return true;
    }
    return false;
}

void Tokenizer::token2Tensor(Net *net, vector<token_id_t> tokens, shared_ptr<Tensor> input_tensor) {
    // auto input_tensor = std::make_shared<Tensor>();
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, 1, static_cast<int>(tokens.size()), 1);
    input_tensor->alloc();
    // input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}

void Tokenizer::tokens2Tensor(Net *net, vector<vector<token_id_t>> tokens, shared_ptr<Tensor> input_tensor) {
    // auto input_tensor = std::make_shared<Tensor>();
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    const auto bsize = static_cast<int>(tokens.size());
    input_tensor->reshape(bsize, 1, static_cast<int>(tokens[0].size()), 1);
    input_tensor->alloc();
    for (int b = 0; b < bsize; ++b) {
        for (int idx = 0; idx < tokens[b].size(); ++idx) {
            input_tensor->setDataAt<float>(b, 0, idx, 0, tokens[b][idx]);
        }
    }
}

// #ifdef ANDROID_API
// void Tokenizer::setAssetManager(AAssetManager *asset_manager) {
//     asset_manager_ = asset_manager;
//     if(!load_vocab(vocab_file_name_)) exit(-1);
//
//
// }
// #endif
Tokenizer::Tokenizer(const std::string &vocab_file) :
    vocab_file_name_(vocab_file) {
    // #ifndef ANDROID_API
    if (!load_vocab(vocab_file)) exit(-1);
    // #endif
}
string Tokenizer::detokenize(const vector<token_id_t> &tokens) {
    // int size = tokens.size() - 1;
    int size = tokens.size();
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
        if (token_id == TokenNl) {
            result += "\n";
            continue;
        }
        result += this->id_token_[token_id].token;
    }
    return result;
}
void Tokenizer::setSpecialToken(const string &bos, const string &eos, const string &unk, const string &nl) {
    if (!bos.empty()) {
        auto bos_token = this->vocab_map_.find(bos);
        if (bos_token != this->vocab_map_.end()) {
            TokenBos = bos_token->second;
        } else {
            MLLM_LOG_ERROR_STREAM << "BOS token not found in vocab file." << std::endl;
        }
    }
    if (!eos.empty()) {
        auto eos_token = this->vocab_map_.find(eos);
        if (eos_token != this->vocab_map_.end()) {
            TokenEos = eos_token->second;
        } else {
            MLLM_LOG_ERROR_STREAM << "EOS token not found in vocab file." << std::endl;
        }
    }
    if (!unk.empty()) {
        auto unk_token = this->vocab_map_.find(unk);
        if (unk_token != this->vocab_map_.end()) {
            TokenUnk = unk_token->second;
        } else {
            MLLM_LOG_ERROR_STREAM << "UNK token not found in vocab file." << std::endl;
        }
    }
}
void Tokenizer::setSpecialTokenMap(std::unordered_map<token_t, token_id_t> special_tokens_map){
    for (auto &special_token : special_tokens_map) {
        auto token = special_token.first;
        auto token_id = special_token.second;
        auto token_id_iter = this->vocab_map_.find(token);
        if (token_id_iter != this->vocab_map_.end()) {
            token_id_iter->second = token_id;
        } else {
            this->vocab_map_[token] = token_id;
            // MLLM_LOG_ERROR_STREAM << "Special token not found in vocab file." << std::endl;
        }
    }
}
} // namespace mllm