//
// Created by Guo Xiaoqiang on 2024/8/12.
//
#ifndef TOKENIZATION_PHI3_HPP
#define TOKENIZATION_PHI3_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <algorithm>
#include <regex>

using namespace mllm;

class Phi3Tokenizer final : public BPETokenizer {
public:
    explicit Phi3Tokenizer(const std::string &vocab_file) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        chat_template_pre = "<|user|>\n";
        chat_template_end = "<|end|>\n<|assistant|>";
    }
    // for phi3v
    std::vector<int> tokenize_vector(const std::string &text) {
        std::string new_text = text;
        bool instruct_f = false;

        new_text = BPETokenizer::replaceString(new_text, ' ', "▁");
        new_text = BPETokenizer::replaceString(new_text, '\n', "▁\n");
        auto parts = _splitWithDelimiters(new_text, special_tokens);

        std::vector<int> tokens_id;
        for (auto &txt : parts) {
            if (std::find(special_tokens.begin(), special_tokens.end(), txt) != special_tokens.end()) {
                std::vector<token_id_t> tmp;
                if (std::find(special_tokens.begin(), special_tokens.end(), txt) != special_tokens.end()) {
                    auto it = this->vocab_map_.find(txt);
                    if (it != this->vocab_map_.end()) {
                        tmp.emplace_back(it->second);
                        // tmp.emplace_back(replaces_token);
                    }
                } else {
                    BPETokenizer::tokenize(txt, tmp, false, true, "");
                }
                tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end());
            } else {
                std::vector<token_id_t> tmp;
                BPETokenizer::tokenize(txt, tmp, false);
                tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end());
            }
        }
        tokens_id.insert(tokens_id.begin(), 1);
        if (!tokens_id.empty() && tokens_id.back() == replaces_token) {
            tokens_id.pop_back();
        }
        return tokens_id;
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        std::string new_text = text;
        bool instruct_f = false;

        new_text = BPETokenizer::replaceString(new_text, ' ', "▁");
        new_text = BPETokenizer::replaceString(new_text, '\n', "▁\n");
        auto parts = _splitWithDelimiters(new_text, special_tokens);

        std::vector<token_id_t> tokens_id;
        for (auto &txt : parts) {
            if (std::find(special_tokens.begin(), special_tokens.end(), txt) != special_tokens.end()) {
                std::vector<token_id_t> tmp;
                if (std::find(special_tokens.begin(), special_tokens.end(), txt) != special_tokens.end()) {
                    auto it = this->vocab_map_.find(txt);
                    if (it != this->vocab_map_.end()) {
                        tmp.emplace_back(it->second);
                        // tmp.emplace_back(replaces_token);
                    }
                } else {
                    BPETokenizer::tokenize(txt, tmp, false, true, "");
                }
                tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end());
            } else {
                std::vector<token_id_t> tmp;
                BPETokenizer::tokenize(txt, tmp, false);
                tokens_id.insert(tokens_id.end(), tmp.begin(), tmp.end());
            }
        }
        tokens_id.insert(tokens_id.begin(), 1);
        if (!tokens_id.empty() && tokens_id.back() == replaces_token) {
            tokens_id.pop_back();
        }
        return BPETokenizer::tokens2Input(tokens_id);
        /*
        if (new_text.rfind(chat_template_pre, 0) == 0 && new_text.find(chat_template_end, new_text.size() - chat_template_end.size()) != std::string::npos) {
            instruct_f = true;
            new_text = new_text.substr(chat_template_pre.size(), new_text.size() - chat_template_pre.size() - chat_template_end.size());
        }

        // replace all blanks with '▁'
        new_text = BPETokenizer::replaceString(new_text, ' ', "▁");

        auto tokens_id = vector<token_id_t>();
        BPETokenizer::tokenize(new_text, tokens_id, false);

        tokens_id.insert(tokens_id.begin(), 1);
        if (instruct_f) {
            // chat template is as follows: <|user|>\n Question <|end|>\n <|assistant|>
            tokens_id.insert(tokens_id.begin() + 1, user_id);
            tokens_id.insert(tokens_id.begin() + 2, 29871);

            tokens_id.insert(tokens_id.begin() + 3, 13);

            tokens_id.insert(tokens_id.end(), end_id);
            tokens_id.insert(tokens_id.end(), 29871);

            tokens_id.insert(tokens_id.end(), 13);
            tokens_id.insert(tokens_id.end(), assistant_id);
        }

        return BPETokenizer::tokens2Input(tokens_id);
        */
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
        if (text == "<|end|>") return {false, ""};
        if (text == "<0x0A>") return {false, ""};
        return {true, text};
    }

public:
    // token_id_t pad_id = 32000, eos_id = 32000, bos_id = 1, user_id = 32010, assistant_id = 32001, end_id = 32007;
    token_id_t replaces_token = 29871; //"▁"
    std::vector<std::string> special_tokens = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|assistant|>",
        "<|system|>",
        "<|end|>",
        "<|user|>"};
};

#endif //! TOKENIZATION_PHI3_HPP
