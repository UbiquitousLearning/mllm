#ifndef TOKENIZATION_BAILING_LITE_HPP
#define TOKENIZATION_BAILING_LITE_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <codecvt>
#include <locale>

using namespace mllm;

class BaiLingTokenizer final : public BPETokenizer {
public:
    explicit BaiLingTokenizer(const std::string &vocab_file, const std::string &merge_file) :
        BPETokenizer(vocab_file) {
        initialize_byte_to_char_map();
        for (const auto &pair : byte_to_char_map_) {
            char_to_byte_map_[pair.second] = pair.first;
        }
        id_to_token_string_.resize(vocab_map_.size() + 1);
        for (const auto &pair : vocab_map_) {
            if (pair.second < id_to_token_string_.size()) {
                id_to_token_string_[pair.second] = pair.first;
            }
        }
        auto merge_file_stream = std::ifstream(merge_file);
        if (!merge_file_stream.good()) {
            std::cout << "merge file is broken\n";
            exit(0);
        }
        std::string line;
        unsigned rank = 0;
        std::unordered_map<std::string, unsigned int> bpe_ranks_;
        std::getline(merge_file_stream, line);
        while (std::getline(merge_file_stream, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            bpe_ranks_[line] = rank;
            rank++;
        }
        BPETokenizer::setMergeRank(bpe_ranks_);
        chat_template_pre = "<role>SYSTEM</role>You are Ling, an assistant created by inclusionAI<role>HUMAN</role>";
        chat_template_end = "<role>ASSISTANT</role>";

        special_tokens_ = {
            bos_token_string_, eos_token_string_, "[CLS]", "[gMASK]",
            "<role>", "</role>",
            "<|arithmetic_start|>", "<|arithmetic_end|>",
            "<|number_start|>", "<|number_end|>"};
        for (int i = 0; i <= 100; ++i) {
            special_tokens_.push_back("<|reserved_token_" + std::to_string(i) + "|>");
        }
        special_tokens_.push_back("<role>SYSTEM</role>");
        special_tokens_.push_back("<role>HUMAN</role>");
        special_tokens_.push_back("<role>BOT</role>");
    }

    Tensor tokenize(const std::string &text, string name = "input_ids", BackendType type = MLLM_CPU) override {
        std::vector<token_id_t> tokens_id;
        auto parts = _splitWithDelimiters(text, special_tokens_);

        for (const auto &part : parts) {
            if (part.empty()) continue;

            auto it = vocab_map_.find(part);
            if (it != vocab_map_.end()) {
                tokens_id.push_back(it->second);
            } else {
                std::string byte_level_string;
                for (unsigned char byte : part) {
                    byte_level_string += u32string_to_utf8({byte_to_char_map_[byte]});
                }
                std::vector<std::string> bpe_pieces = BPETokenizer::bpe(byte_level_string, "");
                for (const auto &piece : bpe_pieces) {
                    auto vocab_it = vocab_map_.find(piece);
                    if (vocab_it != vocab_map_.end()) {
                        tokens_id.push_back(vocab_it->second);
                    } else {
                        std::cerr << "Fatal Error: BPE piece not found in vocab_map_: " << piece << std::endl;
                    }
                }
            }
        }
        return Tokenizer::tokens2Input(tokens_id, name, type);
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        std::string byte_chars_str;
        for (token_id_t token_id : tokens) {
            if (token_id < id_to_token_string_.size()) {
                byte_chars_str += id_to_token_string_[token_id];
            }
        }
        std::u32string u32_byte_chars_str = utf8_to_u32string(byte_chars_str);
        std::vector<char> byte_buffer;
        for (char32_t c : u32_byte_chars_str) {
            auto it = char_to_byte_map_.find(c);
            if (it != char_to_byte_map_.end()) {
                byte_buffer.push_back(static_cast<char>(it->second));
            }
        }
        return std::string(byte_buffer.begin(), byte_buffer.end());
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            scores.push_back(result.dataAt<float>(0, 0, result.sequence() - 1, i));
        }
        auto token_idx = this->argmax(scores);
        return {this->detokenize({token_idx}), token_idx};
    }

    std::pair<bool, std::string> postprocess(std::string &text) override {
        if (text == this->eos_token_string_) return {false, ""};
        if (text == "<role>" || text.rfind("<role>", 0) == 0) {
            return {false, ""};
        }
        if (text == this->bos_token_string_ || text == "<role>" || text == "</role>" || text.rfind("<|reserved_token_", 0) == 0 || text.rfind("<role>", 0) == 0) return {true, ""};
        return {true, text};
    }

private:
    const std::string bos_token_string_ = "<|startoftext|>";
    const std::string eos_token_string_ = "<|endoftext|>";
    std::vector<std::string> special_tokens_;

    std::unordered_map<unsigned char, char32_t> byte_to_char_map_;
    std::unordered_map<char32_t, unsigned char> char_to_byte_map_;
    std::vector<std::string> id_to_token_string_;

    std::vector<std::string> _splitWithDelimiters(const std::string &str, const std::vector<std::string> &delimiters) const {
        std::vector<std::string> result;
        size_t last = 0;
        while (last < str.size()) {
            size_t min_pos = std::string::npos;
            std::string best_delim;
            for (const auto &delim : delimiters) {
                if (!delim.empty()) {
                    size_t found_pos = str.find(delim, last);
                    if (found_pos != std::string::npos && (min_pos == std::string::npos || found_pos < min_pos)) {
                        min_pos = found_pos;
                        best_delim = delim;
                    }
                }
            }
            if (min_pos != std::string::npos) {
                if (min_pos > last) result.push_back(str.substr(last, min_pos - last));
                result.push_back(best_delim);
                last = min_pos + best_delim.length();
            } else {
                result.push_back(str.substr(last));
                break;
            }
        }
        return result;
    }

    static std::u32string utf8_to_u32string(const std::string &s) {
        try {
            std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
            return conv.from_bytes(s);
        } catch (const std::range_error &) { return {}; }
    }
    static std::string u32string_to_utf8(const std::u32string &s) {
        try {
            std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
            return conv.to_bytes(s);
        } catch (const std::range_error &) { return ""; }
    }

    void initialize_byte_to_char_map() {
        std::vector<char32_t> chars;
        for (int i = 0; i < 256; ++i) { chars.push_back(static_cast<char32_t>(i)); }
        int n = 0;
        for (int i = 0; i < 256; ++i) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
                chars[i] = 256 + n;
                n++;
            }
        }
        for (int i = 0; i < 256; ++i) {
            byte_to_char_map_[static_cast<unsigned char>(i)] = chars[i];
        }
    }
};

#endif // TOKENIZATION_BAILING_LITE_HPP