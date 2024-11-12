//
// Created by Xiang Li on 2023/12/2.
//
#include <algorithm>
#include "Unigram.hpp"
#include "sstream"
using namespace mllm;

UnigramTokenizer::UnigramTokenizer(const std::string &vocab_file) :
    Tokenizer(std::move(vocab_file)) {
    for (auto &item : this->id_token_) {
        std::vector<char> vec(item.token.begin(), item.token.end());
        trie_.insert(vec);
    }
}

void UnigramTokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos, bool byte_fallback) {
    if (text.empty()) {
        return;
    }
    if (bos) {
        tokens.emplace_back(TokenBos);
    }

    auto size = text.size();
    std::vector<BestPath> best_path(size + 1);
    auto unk_score = this->min_score_ - K_UNK_PENALTY;
    auto starts_at = 0;
    while (starts_at < size) {
        auto best_path_score = best_path[starts_at].best_path_score;
        bool has_single_char = false;
        auto sub_str = text.substr(starts_at);
        auto mblen = 1;
        std::vector<char> vec(sub_str.begin(), sub_str.end());
        auto sub_len = sub_str.length();
        auto iter = trie_.commonPrefixSearch(vec);
        auto item = iter->next();
        // for (std::vector<char> item = iter.next(); !item.empty(); item = iter.next()) {
        while (!item.empty()) {
            auto key_pos = starts_at + item.size();
            auto token = std::string(item.begin(), item.end());
            auto token_id = this->vocab_map_.find(token);
            auto &target_node = best_path[key_pos];
            if (token_id != this->vocab_map_.end()) {
                auto score = this->id_token_[token_id->second].score;
                auto new_score = best_path_score + score;
                if (new_score > target_node.best_path_score || target_node.starts_at == -1) {
                    target_node.best_path_score = new_score;
                    target_node.starts_at = starts_at;
                    target_node.id = token_id->second;
                }
                if (!has_single_char && item.size() == mblen) {
                    has_single_char = true;
                }
            }
            item = iter->next();
        }
        if (!has_single_char) {
            auto &target_node = best_path[starts_at + mblen];
            auto new_score = best_path_score + unk_score;
            if (new_score > target_node.best_path_score || target_node.starts_at == -1) {
                target_node.best_path_score = new_score;
                target_node.starts_at = starts_at;
                target_node.id = TokenUnk;
            }
        }
        starts_at += mblen;
    }
    auto ends_at = size;
    std::vector<std::string> result;
    //    std::vector<std::string> token_;
    std::string token_;
    while (ends_at > 0) {
        auto target_node = best_path[ends_at];
        //        starts_at = target_node.starts_at;
        auto sub_str = text.substr(target_node.starts_at, ends_at - target_node.starts_at);

        if (target_node.id == TokenUnk) {
            token_.append(sub_str);
        } else {
            if (!token_.empty()) {
                std::reverse(token_.begin(), token_.end());
                result.emplace_back(token_);
                token_.clear();
            }
            result.emplace_back(sub_str);
        }
        ends_at = target_node.starts_at;
    }
    if (!token_.empty()) {
        std::reverse(token_.begin(), token_.end());
        result.emplace_back(token_);
        token_.clear();
    }
    std::reverse(result.begin(), result.end());

    for (auto &item : result) {
        auto token_id = this->vocab_map_.find(item);
        if (token_id != this->vocab_map_.end()) {
            tokens.emplace_back(token_id->second);
        } else {
            if (byte_fallback) {
                for (char j : item) {
                    char *byte_string = new char[10];
                    sprintf(byte_string, "<0x%02X>", j);
                    auto result = this->vocab_map_.find(byte_string);
                    if (result != this->vocab_map_.end()) {
                        tokens.emplace_back(result->second);
                    }else {
                        MLLM_LOG_ERROR_STREAM << "byte_fallback error" << byte_string << std::endl;
                    }
                }
            }else tokens.emplace_back(TokenUnk);
        }
    }
}

std::string UnigramTokenizer::detokenize(const std::vector<token_id_t> &tokens) {
    int size = tokens.size();
    std::string result;
    for (int i = 0; i < size; i++) {
        auto token_id = tokens[i];
        if (token_id == TokenUnk) {
            result += "<unk>";
            continue;
        }
        if (token_id == TokenBos) {
            continue;
        }
        if (token_id == TokenEos) {
            if (i != size - 1) {
                result += " ";
            }
        }
        auto token  = this->id_token_[token_id].token;
        if (token[0] == '<' && token[token.size() - 1] == '>') {
            std::stringstream ss;
            ss << std::hex << token.substr(3, token.size() - 4);
            int n;
            ss >> n;
            result += static_cast<char>(n);
            // replace ▁[wide char] with " "
            //TODO:Fuyu only
        } else if ((int)token[0] == -30 && (int)token[1] == -106 && (int)token[2] == -127) {
            // if (i != size - 1) {
                result += " ";
            // }
            if (token.size() > 3) {
                result += token.substr(3);
            }
        }else {
            result += token;
        }

    }
    return result;
}

void UnigramTokenizer::tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) {
    this->tokenize(std::move(text), tokens, bos, true);
}
