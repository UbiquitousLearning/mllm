#ifndef TOKENIZATION_MINICPM_HPP
#define TOKENIZATION_MINICPM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <regex>

using namespace mllm;

class MiniCPMTokenizer final : public BPETokenizer {
    std::unordered_map<std::string, unsigned> merge_rank;
    token_id_t eos_id = 2, bos_id = 1;
    std::string token_start = "<s>", token_end = "</s>";
    std::string token_unkonw = "<unk>";

public:
    explicit MiniCPMTokenizer(const std::string &vocab_file, const std::string &merge_file) :
        BPETokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        std::ifstream merge(merge_file);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank++;
        }
        BPETokenizer::setMergeRank(merge_rank);
        BPETokenizer::setSpecialToken(token_unkonw, token_start, token_end);
        chat_template_pre = "<用户>";
        chat_template_end = "<AI>";
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        auto new_text = " " + text;
        new_text = std::regex_replace(new_text, std::regex(" "), "▁");

        std::vector<token_id_t> tokens_id;
        BPETokenizer::tokenize(new_text, tokens_id, true);
        tokens_id[0] = bos_id;
        tokens_id.pop_back();
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
        if (text == "<0x0A>") return {true, "\n"};
        if (text == "</s>") return {false, ""};
        if (text == "<unk>") return {false, ""};
        return {true, text};
    }
};

#endif // TOKENIZATION_MINICPM_HPP