//
// Created by xwk on 24-10-26.
//

#ifndef MLLM_WORDPIECE_HPP
#define MLLM_WORDPIECE_HPP

#include "tokenizers/Tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <regex>
#include <locale>
#include <codecvt>
#include <cctype>
#include <algorithm>

namespace mllm{

class BasicTokenizer {
public:
    BasicTokenizer(bool do_lower_case = true,
                   std::vector<std::wstring> const& never_split = {},
                   bool tokenize_chinese_chars = true,
                   bool strip_accents = true,
                   bool do_split_on_punc = true)
        : do_lower_case(do_lower_case),
        _tokenize_chinese_chars(tokenize_chinese_chars),
        strip_accents(strip_accents),
        do_split_on_punc(do_split_on_punc),
        never_split(never_split.begin(), never_split.end()) {}

    std::vector<std::wstring> tokenize(const std::wstring& text);

    void add_never_split(const std::wstring& token);

private:
    bool do_lower_case;
    bool _tokenize_chinese_chars;
    bool strip_accents;
    bool do_split_on_punc;
    std::unordered_set<std::wstring> never_split;

    std::wstring clean_text(const std::wstring& text);
    std::wstring strip_accents_from_text(const std::wstring& input);
    std::vector<std::wstring> split_on_punctuation(const std::wstring& text);
    std::wstring tokenize_chinese_chars(const std::wstring& text);
    bool is_chinese_char(wchar_t cp);
    static bool is_punctuation(wchar_t ch);
};


class WordPieceTokenizer: public Tokenizer {
public:
    BasicTokenizer basic_tokenizer;

    WordPieceTokenizer(const std::string &vocab_file);
    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, bool bos) override;

    void add_special_tokens(const std::vector<std::string> &special_tokens);
};

}

#endif // MLLM_WORDPIECE_HPP
