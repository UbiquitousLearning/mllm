#ifndef TOKENIZATION_STABLELM_HPP
#define TOKENIZATION_STABLELM_HPP

#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;

class stablelmTokenizer final {
    BPETokenizer *tokenizer;
    std::unordered_map<std::string, unsigned> merge_rank;

    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        unsigned int maxIndex = 0;
        float maxValue = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > maxValue) {
                maxIndex = i;
                maxValue = scores[i];
            }
        }
        return maxIndex;
    }

public:
    explicit stablelmTokenizer(const std::string &vocab_file, const std::string &merge_file) {
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
        tokenizer->setSpecialToken("<|endoftext|>", "<|im_end|>", "<|endoftext|>");
    }

    Tensor tokenize(std::string &text, int str_i = 0) const {
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        text = Tokenizer::replaceString(text, ' ', "Ġ");
        std::vector<token_id_t> tokens_id;
        tokenizer->tokenize(text, tokens_id, true);
        tokens_id.erase(tokens_id.begin());
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
};

#endif // TOKENIZATION_STABLELM_HPP