//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef TOKENIZATION_LLAMA_HPP
#define TOKENIZATION_LLAMA_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <regex>

using namespace mllm;

class LLaMATokenizer final : public BPETokenizer {
    bool bos_ = true;

public:
    explicit LLaMATokenizer(const std::string &vocab_file, bool bos = true) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        bos_ = bos;
        chat_template_pre = "<s>[INST] ";
        chat_template_end = " [/INST]";
    }
    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        auto tokens_id = vector<token_id_t>();
        BPETokenizer::tokenize(text, tokens_id, bos_);
        return tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        return BPETokenizer::detokenize(tokens);
    }
    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = argmax(scores);
        return {BPETokenizer::detokenize({token_idx}), token_idx};
    }
    std::pair<bool, std::string> postprocess(std::string &text) override {
        text = std::regex_replace(text, std::regex("▁"), " ");
        if (text.empty()) return {false, ""};
        if (text == "<|endoftext|>" || text == "<|im_end|>") return {false, ""};
        return {true, text};
    }
};

#endif // TOKENIZATION_LLAMA_HPP
