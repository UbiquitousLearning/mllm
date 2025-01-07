// MIT License

// Copyright (c) 2023-2024 The ggml authors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
#pragma once

#include <cstdint>
#include <locale>
#include <string>
#include <vector>
// unicode
#include <codecvt>

// TODO: prefix all symbols with "llama_"
struct codepoint_flags {
    enum {
        UNDEFINED = 0x0001,
        NUMBER = 0x0002,      // regex: \p{N}
        LETTER = 0x0004,      // regex: \p{L}
        SEPARATOR = 0x0008,   // regex: \p{Z}
        ACCENT_MARK = 0x0010, // regex: \p{M}
        PUNCTUATION = 0x0020, // regex: \p{P}
        SYMBOL = 0x0040,      // regex: \p{S}
        CONTROL = 0x0080,     // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
    };

    // codepoint type
    uint16_t is_undefined : 1;
    uint16_t is_number : 1;      // regex: \p{N}
    uint16_t is_letter : 1;      // regex: \p{L}
    uint16_t is_separator : 1;   // regex: \p{Z}
    uint16_t is_accent_mark : 1; // regex: \p{M}
    uint16_t is_punctuation : 1; // regex: \p{P}
    uint16_t is_symbol : 1;      // regex: \p{S}
    uint16_t is_control : 1;     // regex: \p{C}
    // helper flags
    uint16_t is_whitespace : 1; // regex: \s
    uint16_t is_lowercase : 1;
    uint16_t is_uppercase : 1;
    uint16_t is_nfd : 1;

    // decode from uint16
    inline codepoint_flags(const uint16_t flags = 0) {
        *reinterpret_cast<uint16_t *>(this) = flags;
    }

    inline uint16_t as_uint() const {
        return *reinterpret_cast<const uint16_t *>(this);
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8(uint32_t cp);
uint32_t unicode_cpt_from_utf8(const std::string &utf8, size_t &offset);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string &utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> &cpts);

codepoint_flags unicode_cpt_flags(uint32_t cp);
codepoint_flags unicode_cpt_flags(const std::string &utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string &utf8);

uint32_t unicode_tolower(uint32_t cp);

std::vector<std::string> unicode_regex_split_naive(const std::string &text, const std::vector<std::string> &regex_exprs);

std::vector<std::string> unicode_regex_split(const std::string &text, const std::vector<std::string> &regex_exprs);

static std::string any_to_utf8(std::string s) {
    // the original input is utf-8 already
    return s;
}

static std::string __chr(int v) {
    std::wstring wString(1, v);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
    std::string utf8 = convert.to_bytes(wString);
    return utf8;
}

static std::vector<int> __ord(std::string v) {
    std::vector<int> ret;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
    std::wstring utf8str = convert.from_bytes(v);
    for (auto i = 0; i < utf8str.length(); ++i) ret.emplace_back(utf8str[i]);
    return ret;
}

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";
static const std::string SPLIT_PAT_STR = R"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)";
static const std::vector<std::string> FIXED_PAT_STRS = {
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
};