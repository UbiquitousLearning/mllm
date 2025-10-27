#ifndef TOKENIZATION_MINICPM_HPP
#define TOKENIZATION_MINICPM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <regex>

using namespace mllm;

class MiniCPM3Tokenizer final : public BPETokenizer {
    std::unordered_map<std::string, unsigned> merge_rank;
    token_id_t eos_id = 2, bos_id = 1;
    std::string token_start = "<s>", token_end = "</s>";
    std::string token_unkonw = "<unk>";
    std::vector<std::string> special_tokens = {
        "<|im_end|>",
        "<|im_start|>",
        "<|tool_call|>",
        "<|execute_start|>",
        "<|execute_end|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        token_start,
        token_end,
        token_unkonw,
        "\n",
        "▁assistant",
    };
    token_id_t replaces_token = 29871; //"▁"
public:
    explicit MiniCPM3Tokenizer(const std::string &vocab_file) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        chat_template_pre = "<|im_start|>▁user\n";
        chat_template_end = "<|im_end|>▁\n<|im_start|>▁assistant\n";
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {

        std::string new_text = text;
        bool instruct_f = false;

        new_text = BPETokenizer::replaceString(new_text, ' ', "▁");
        // new_text = BPETokenizer::replaceString(new_text, '\n', "▁\n");
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
        // tokens_id.insert(tokens_id.begin(), 1);
        // if (!tokens_id.empty() && tokens_id.back() == replaces_token) {
        //     tokens_id.pop_back();
        // }
        // tokens_id.pop_back();
        return BPETokenizer::tokens2Input(tokens_id);

    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        return BPETokenizer::detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1 && result.head() == 1);
        std::vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {BPETokenizer::detokenize({token_idx}), token_idx};
    }
    std::pair<bool, std::string> postprocess(std::string &text) override {
        text = std::regex_replace(text, std::regex("▁"), " ");
        if (text == "<|im_end|>") return {false, ""};
        if (text == "</s>") return {false, ""};
        if (text == "<unk>") return {false, ""};
        return {true, text};
    }
};

#endif // TOKENIZATION_MINICPM_HPP