#ifndef TOKENIZATION_MINICPM_HPP
#define TOKENIZATION_MINICPM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <regex>

using namespace mllm;

class MiniCPMTokenizer final {
    BPETokenizer *tokenizer;
    std::unordered_map<std::string, unsigned> merge_rank;

    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

public:
    explicit MiniCPMTokenizer(const std::string &vocab_file, const std::string &merge_file) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);
        std::ifstream merge(merge_file);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank++;
        }
        tokenizer->setMergeRank(merge_rank);
        tokenizer->setSpecialToken("<unk>", "<s>", "</s>");
    }

    Tensor tokenize(std::string &text) const {
        auto new_text = " " + text;
        new_text = std::regex_replace(new_text, std::regex(" "), "▁");

        std::vector<token_id_t> tokens_id;
        tokenizer->tokenize(new_text, tokens_id, true);
        tokens_id[0] = bos_id;
        tokens_id.pop_back();
        return BPETokenizer::tokens2Input(tokens_id);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1 && result.head() == 1);
        std::vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {tokenizer->detokenize({token_idx}), token_idx};
    }

    token_id_t eos_id = 2, bos_id = 1;
    std::string token_user_o = "<用户>", token_user_c = "<AI>";
    std::string token_start = "<s>", token_end = "</s>";
    std::string token_unkonw = "<unk>";
};

#endif // TOKENIZATION_MINICPM_HPP