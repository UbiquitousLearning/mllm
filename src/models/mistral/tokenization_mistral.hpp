/**
 * @file tokenization_mistral.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Mistral tokenization method
 * @version 0.1
 * @date 2024-05-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_MISTRAL_HPP
#define TOKENIZATION_MISTRAL_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <regex>

using namespace mllm;

class MistralTokenizer final : public BPETokenizer {
public:
    explicit MistralTokenizer(const std::string &vocab_file) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        chat_template_pre = "<s>[INST] ";
        chat_template_end = " [/INST]";
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        // auto newText = token_start + token_user_o + " " + text + token_user_c + token_end;
        auto tokens_id = vector<token_id_t>();
        BPETokenizer::tokenize(text, tokens_id, false);
        return BPETokenizer::tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        return BPETokenizer::detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(result.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        std::vector<float> scores;
        int _dims = result.dimension();
        int _seq = result.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = result.dataAt<float>(0, 0, _seq, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        auto text = BPETokenizer::detokenize({token_idx});
        return make_pair(text, token_idx);
    }
    std::pair<bool, std::string> postprocess(std::string &text) override {
        text = std::regex_replace(text, std::regex("▁"), " ");
        if (text == "<0x0A>") return {true, "\n"};
        if (text == "</s>") return {false, ""};
        return {true, text};
    }

public:
    token_id_t eos_id = 2, bos_id = 1;
    std::string token_user_o = "[INST]", token_user_c = "[/INST]";
    std::string token_start = "<s>", token_end = "</s>";
    std::string token_unkonw = "<unk>";
};

#endif //! TOKENIZATION_MISTRAL_HPP