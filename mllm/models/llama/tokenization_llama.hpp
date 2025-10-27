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
    std::vector<std::string> _splitWithDelimiters(const std::string &str, const std::vector<std::string> &delimiters) {
        std::vector<std::string> result;
        size_t pos = 0, last = 0;
        while (pos < str.size()) {
            size_t match_pos = std::string::npos;
            size_t match_len = 0;
            for (const auto &delim : delimiters) {
                if (!delim.empty() && str.compare(pos, delim.size(), delim) == 0) {
                    match_pos = pos;
                    match_len = delim.size();
                    break;
                }
            }
            if (match_pos != std::string::npos) {
                if (pos > last)
                    result.push_back(str.substr(last, pos - last));
                result.push_back(str.substr(pos, match_len));
                pos += match_len;
                last = pos;
            } else {
                ++pos;
            }
        }
        if (last < str.size())
            result.push_back(str.substr(last));
        return result;
    }
    bool hf_f = false;

public:
    explicit LLaMATokenizer(const std::string &vocab_file, bool bos = true) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        bos_ = bos;
        chat_template_pre = "<s><s> [INST] ";
        chat_template_end = " [/INST]"; // 判断 vocab_file 是否包含 "hf"
        if (vocab_file.find("hf") != std::string::npos) {
            hf_f = true;
        }
    }
    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        auto tokens_id = vector<token_id_t>();
        if (hf_f) {
            string n_text = std::regex_replace(text, std::regex(" "), "▁");
            std::vector<std::string> special_tokens = {
                "<s>",
                "INST"};
            auto parts = _splitWithDelimiters(n_text, special_tokens);
            for (auto &p : parts) {
                if (std::find(special_tokens.begin(), special_tokens.end(), p) != special_tokens.end()) {
                    std::string token;
                    for (auto b : p) token += b;
                    std::vector<token_id_t> tmp;
                    BPETokenizer::tokenize(token, tmp, false, special_tokens, true);
                    if (tmp.empty()) continue;
                    tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end() - 1);
                } else {
                    std::vector<token_id_t> tmp;
                    BPETokenizer::tokenize(p, tmp, false, true, "");
                    tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end());
                }
            }
            return tokens2Input(tokens_id);
        }
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
        if (text == "<0x0A>") return {true, "\n"};
        if (text.empty()) return {false, ""};
        if (text == "<|endoftext|>" || text == "<|im_end|>") return {false, ""};
        return {true, text};
    }
};

#endif // TOKENIZATION_LLAMA_HPP
