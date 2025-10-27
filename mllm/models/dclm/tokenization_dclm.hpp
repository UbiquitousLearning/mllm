/**
 * @file tokenization_dclm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef TOKENIZATION_DCLM_HPP
#define TOKENIZATION_DCLM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <algorithm>

using namespace mllm;

using namespace mllm;

class DCLMTokenizer final : public BPETokenizer {
    std::unordered_map<std::string, unsigned> merge_rank;

public:
    explicit DCLMTokenizer(const std::string &vocab_file, const std::string &merge_file) :
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
        BPETokenizer::setSpecialToken("<|endoftext|>", "<|endoftext|>", "<|endoftext|>");
    }
    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        auto new_text = Tokenizer::replaceString(text, ' ', "Ġ");
        std::vector<token_id_t> tokens_id;
        BPETokenizer::tokenize(new_text, tokens_id, false);
        return BPETokenizer::tokens2Input(tokens_id);
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
        auto token_idx = this->argmax(scores);
        return {BPETokenizer::detokenize({token_idx}), token_idx};
    }

    std::pair<bool, std::string> postprocess(std::string &text) override {
        size_t pos = 0;
        while ((pos = text.find("Ċ", pos)) != std::string::npos) {
            text.replace(pos, 2, " ");
            break;
        }
        pos = 0;
        while ((pos = text.find("Ġ", pos)) != std::string::npos) {
            text.replace(pos, 2, " ");
        }
        return {true, text};
    }
};

#endif //! TOKENIZATION_DCLM_HPP