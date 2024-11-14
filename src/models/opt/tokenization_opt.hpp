#ifndef TOKENIZATION_OPT_HPP
#define TOKENIZATION_OPT_HPP

#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;

class OPTTokenizer final : public BPETokenizer {
    std::unordered_map<std::string, unsigned> merge_rank;

public:
    explicit OPTTokenizer(const std::string &vocab_file, const std::string &merge_file) :
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
        BPETokenizer::setSpecialToken("</s>", "");
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        string new_text = text;
        if (text[0] != ' ') {
            new_text = ' ' + text;
        }
        new_text = Tokenizer::replaceString(new_text, ' ', "Ġ");
        std::vector<token_id_t> tokens_id;
        BPETokenizer::tokenize(new_text, tokens_id, true);

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
        size_t pos = 0;
        while ((pos = text.find("Ċ", pos)) != std::string::npos) {
            text.replace(pos, 2, " ");
        }
        pos = 0;
        while ((pos = text.find("Ġ", pos)) != std::string::npos) {
            text.replace(pos, 2, " ");
        }
        if (text == "</s>") return {false, ""};
        return {true, text};
    }
};

#endif // TOKENIZATION_OPT_HPP