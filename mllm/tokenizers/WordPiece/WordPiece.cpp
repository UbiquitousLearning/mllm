//
// Created by xwk on 24-10-26.
//

#include "WordPiece.hpp"

std::wstring utf8_to_wstring(const std::string& utf8str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(utf8str);
}

std::string wstring_to_utf8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

std::wstring trim(const std::wstring &s) {
    auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c) { return std::isspace(c); });
    auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) { return std::isspace(c); }).base();
    return (wsback <= wsfront ? std::wstring() : std::wstring(wsfront, wsback));
}

std::vector<std::wstring> split(const std::wstring &s) {
    std::wistringstream iss(s);
    std::vector<std::wstring> tokens;
    std::wstring token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::wstring> whitespace_tokenize(const std::wstring &text) {
    std::wstring trimmed_text = trim(text);
    if (trimmed_text.empty()) {
        return {};
    }
    return split(trimmed_text);
}

mllm::WordPieceTokenizer::WordPieceTokenizer(const string &vocab_file) :
    Tokenizer(vocab_file), basic_tokenizer(true) {
}

bool mllm::BasicTokenizer::is_punctuation(wchar_t ch) {
    // Simplified check assuming punctuation falls within these ranges
    return std::ispunct(ch) != 0;
}

std::wstring mllm::BasicTokenizer::clean_text(const std::wstring& text) {
    std::wstring output;
    for (wchar_t c : text) {
        if (c == 0 || c == 0xFFFD || std::iscntrl(c)) continue;
        if (std::iswspace(c))
            output += L' ';
        else
            output += c;
    }
    return output;
}

std::wstring mllm::BasicTokenizer::strip_accents_from_text(const std::wstring& input) {
    // This function needs proper implementation depending on the locale
    return input;  // Simplified as placeholder
}

std::vector<std::wstring> mllm::BasicTokenizer::split_on_punctuation(const std::wstring& text) {
    std::vector<std::wstring> result;
    std::wstring token;
    for (wchar_t ch : text) {
        if (is_punctuation(ch)) {
            if (!token.empty()) {
                result.push_back(token);
                token.clear();
            }
            result.push_back(std::wstring(1, ch));
        } else {
            token += ch;
        }
    }
    if (!token.empty())
        result.push_back(token);
    return result;
}

std::wstring mllm::BasicTokenizer::tokenize_chinese_chars(const std::wstring& text) {
    std::wstring output;
    for (wchar_t ch : text) {
        if (is_chinese_char(ch)) {
            output += L' ';
            output += ch;
            output += L' ';
        } else {
            output += ch;
        }
    }
    return output;
}

bool mllm::BasicTokenizer::is_chinese_char(wchar_t cp) {
    // Simplified check for Chinese characters range
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF);
}

std::vector<std::wstring> splitBySet(const std::wstring& text, const std::unordered_set<std::wstring>& words) {
    std::vector<std::wstring> result;
    size_t pos = 0;

    while (pos < text.length()) {
        size_t minPos = std::wstring::npos;
        std::wstring foundWord;

        // 查找最近的匹配项
        for (const auto& word : words) {
            size_t found = text.find(word, pos);
            if (found != std::wstring::npos && (found < minPos)) {
                minPos = found;
                foundWord = word;
            }
        }

        // 如果找到匹配项，处理之前的文本和匹配项
        if (minPos != std::wstring::npos) {
            if (minPos > pos) {
                // 添加匹配项前的文本
                result.push_back(text.substr(pos, minPos - pos));
            }
            // 添加匹配项
            result.push_back(foundWord);
            pos = minPos + foundWord.size();
        } else {
            // 没有更多匹配项，添加剩余所有文本
            result.push_back(text.substr(pos));
            break;
        }
    }

    return result;
}

std::vector<std::wstring> mllm::BasicTokenizer::tokenize(const std::wstring& text) {
    std::wstring cleaned = clean_text(text);
    if (_tokenize_chinese_chars)
        cleaned = tokenize_chinese_chars(cleaned);
    std::vector<std::wstring> white_space_splited_tokens = whitespace_tokenize(cleaned);
    std::vector<std::wstring> split_tokens;
    for (const auto& token : white_space_splited_tokens) {
        auto sub_tokens = splitBySet(token, never_split);
        split_tokens.insert(split_tokens.end(), sub_tokens.begin(), sub_tokens.end());
    }
    std::vector<std::wstring> output_tokens;

    for (auto& token : split_tokens) {
//        std::cout << "token: " << wstring_to_utf8(token) << std::endl;
        if (never_split.count(token)) {
            output_tokens.push_back(token);
        }else{
            if (do_lower_case) {
                std::transform(token.begin(), token.end(), token.begin(),
                               [](wchar_t c){ return std::towlower(c); });
            }

            if (strip_accents)
                token = strip_accents_from_text(token);

            // split on punctuation and add to output
            auto sub_tokens = split_on_punctuation(token);
//            std::cout << "token: " << wstring_to_utf8(token) << std::endl;
            for (auto& sub_token : sub_tokens) {
                if (!sub_token.empty()) {
//                    std::cout << "sub_token: " << wstring_to_utf8(sub_token) << std::endl;
                    output_tokens.push_back(sub_token);
                }
            }
        }
    }

    return output_tokens;
}
void mllm::BasicTokenizer::add_never_split(const std::wstring &token) {
    never_split.insert(token);
}

void mllm::WordPieceTokenizer::tokenize(const string &text, vector<token_id_t> &tokens, bool bos) {
    auto wstr = utf8_to_wstring(text);
    auto basic_tokens = basic_tokenizer.tokenize(wstr);
    auto token_strs = vector<string> ();
    auto token_ids = vector<token_id_t>();
    for (const auto& token : basic_tokens) {
        int start = 0;
        while(start < token.size()) {
            auto end = token.size();
            string str;
            while(start < end){
                auto sub_str = token.substr(start, end - start);
                if (start > 0)
                    sub_str = L"##" + sub_str;
                auto utf8_str = wstring_to_utf8(sub_str);
//                std::cout << "utf8_str: " << utf8_str << std::endl;
                if (vocab_map_.count(utf8_str)){
                    str = utf8_str;
                    break;
                }else{
                    end--;
                }
            }
            if (str.empty()){
                token_strs.push_back("[UNK]");
                break;
            } else{
                token_strs.push_back(str);
//                printf("word: %s\n", str.c_str());
            }
            start = end;
        }
    }

    for (const auto& token_str : token_strs) {
//        std::cout << "token: " << token_str << std::endl;
        tokens.push_back(vocab_map_[token_str]);
    }
}
void mllm::WordPieceTokenizer::add_special_tokens(const vector<std::string> &special_tokens) {
    // add never split tokens to basic tokenizer
    for (const auto& token : special_tokens) {
        basic_tokenizer.add_never_split(utf8_to_wstring(token));
    }
}
