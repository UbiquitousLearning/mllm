//
// Created by Rongjie Yi on 24-3-8.
//

#ifndef PROCESSING_LLAVA_HPP
#define PROCESSING_LLAVA_HPP
#include "tokenizers/BPE/Bpe.hpp"
#include "models/clip/processing_clip.hpp"
#include <utility>
#include <array>

using namespace mllm;

class LLaVAProcessor final : public ClipProcessor {
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
    explicit LLaVAProcessor(const string &vocab_path, const string &merges_path, int height = 336, int width = 336) :
        ClipProcessor(vocab_path, merges_path, height, width, false) {
        Module::initBackend(MLLM_CPU);
    }

    std::array<Tensor, 2> process(string text, string img_path, int hw = 336,
                                  string img_name = "input_vision", string text_name = "input_text", BackendType type = MLLM_CPU) {
        input_ids_.clear();
        pixel_values_.clear();
        auto tokens_ids = vector<vector<token_id_t>>();
        // if (text[0] != ' ') {
        //     text = ' ' + text;
        // }
        vector<mllm::token_id_t> tokens_id = {};
        tokenizer->tokenize(BPETokenizer::replaceString(text, ' ', "▁"), tokens_id, {"<image>", "<pad>", "\n"});
        tokens_ids.push_back(tokens_id);
        PreProcessImages({std::move(img_path)}, hw, hw);
        auto images = pixel_values_[0];

        return {Tokenizer::tokens2Input(tokens_ids, std::move(text_name)), img2Tensor(images, std::move(img_name))};
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {tokenizer->detokenize({token_idx}), token_idx};
    }
    std::pair<bool, std::string> postprocess(std::string &text) {
        size_t pos = 0;
        std::string from = "▁";
        std::string to = " ";
        while ((pos = text.find(from, pos)) != std::string::npos) {
            text.replace(pos, from.length(), to);
            pos += to.length();
        }
        if (text == "</s>") return {false, ""};
        return {true, text};
    }
};

#endif // PROCESSING_LLAVA_HPP
