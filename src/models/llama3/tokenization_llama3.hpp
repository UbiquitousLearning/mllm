//
// Created by xwk on 25-1-10.
//

#ifndef TOKENIZATION_LLAMA3_HPP
#define TOKENIZATION_LLAMA3_HPP

// #include <iostream>
#include <ctime>

#include "tokenizers/Tiktoken/tiktoken.hpp"
// #include <regex>

using namespace mllm;

class LLama3Tokenizer final : public TiktokenTokenizer {
public:
    // explicit LLama3Tokenizer(const merge_rank_t &mergeable_ranks, const unordered_map<std::string, rank_t> &specialTokensEncoder, const string &pattern) :
    //     TiktokenTokenizer(mergeable_ranks, specialTokensEncoder, pattern) {
    //     chat_template_pre = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    //                         "\n"
    //                         "Cutting Knowledge Date: December 2023\n"
    //                         + _getCurrentDateString() + "\n"
    //                                                     "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    //                                                     "";
    //     chat_template_end = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    // }
    explicit LLama3Tokenizer(const std::string &filename) :
        TiktokenTokenizer(filename) {
        const string pattern = R"del((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)del";

        auto mergable_ranks = load_tiktoken_bpe(filename);

        unordered_map<string, rank_t> special_tokens_map;

        // 构造 special_tokens 列表
        std::vector<std::string> special_tokens_string = {
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>", // end of turn
        };

        // 动态生成的 token
        int num_reserved_special_tokens = 256;
        for (int i = 5; i < num_reserved_special_tokens - 5; ++i) {
            special_tokens_string.push_back("<|reserved_special_token_" + std::to_string(i) + "|>");
        }

        for (size_t i = 0; i < special_tokens_string.size(); ++i) {
            special_tokens_map[special_tokens_string[i]] = mergable_ranks.size() + i;
        }
        core = CoreBPE(mergable_ranks, special_tokens_map, pattern);
        for (auto &pair : special_tokens_map) {
            special_tokens.insert(pair.first);
        }

        chat_template_pre = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                            "\n"
                            "Cutting Knowledge Date: December 2023\n"
                            + _getCurrentDateString() + "\n"
                                                        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                                                        "";
        chat_template_end = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    Tensor tokenize(const std::string &text, string name = "input", BackendType type = MLLM_CPU) override {
        auto tokens_id = vector<token_id_t>();
        TiktokenTokenizer::tokenize(text, tokens_id, false);
        return tokens2Input(tokens_id, name, type);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        return TiktokenTokenizer::detokenize(tokens);
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
        return {TiktokenTokenizer::detokenize({token_idx}), token_idx};
    }

    std::pair<bool, std::string> postprocess(std::string &text) override {
        if (text.empty()) return {false, ""};
        if (text == "<|endoftext|>" || text == "<|eot_id|>") return {false, ""};
        return {true, text};
    }

    static std::string _getCurrentDateString() {
        std::time_t now = std::time(nullptr);
        std::tm *local_time = std::localtime(&now);

        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "Today Date: %d %b %Y\n", local_time);

        return std::string(buffer);
    }
};

#endif // TOKENIZATION_LLAMA3_HPP
