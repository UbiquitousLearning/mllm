/*
llama.cpp - commit 54ef9cfc
https://github.com/ggerganov/llama.cpp

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Hash function for std::pair<uint32_t, uint32_t> used in composition table
namespace std {
template<>
struct hash<std::pair<uint32_t, uint32_t>> {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
    return std::hash<uint64_t>{}(((uint64_t)p.first << 32) | p.second);
  }
};
}  // namespace std

size_t unicode_len_utf8(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

// Unused function
// static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> &cps) {
//   std::string result;
//   for (size_t i = 0; i < cps.size(); ++i) {
//     result.append(unicode_cpt_to_utf8(cps[i]));
//   }
//   return result;
// }

uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset) {
  assert(offset < utf8.size());
  if (!(utf8[offset + 0] & 0x80)) {
    auto result = utf8[offset + 0];
    offset += 1;
    return result;
  }
  if (!(utf8[offset + 0] & 0x40)) { throw std::invalid_argument("invalid character"); }
  if (!(utf8[offset + 0] & 0x20)) {
    if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) { throw std::invalid_argument("invalid character"); }
    auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
    offset += 2;
    return result;
  }
  if (!(utf8[offset + 0] & 0x10)) {
    if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
    offset += 3;
    return result;
  }
  if (!(utf8[offset + 0] & 0x08)) {
    if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)
        || !((utf8[offset + 3] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6)
                  | (utf8[offset + 3] & 0x3f);
    offset += 4;
    return result;
  }
  throw std::invalid_argument("failed to convert utf8 to codepoint");
}

// static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cp) {
//     std::vector<uint16_t> result;
//     if (/* 0x0000 <= cp && */ cp <= 0xffff) {
//         result.emplace_back(cp);
//         return result;
//     }
//     if (0x10000 <= cp && cp <= 0x10ffff) {
//         result.emplace_back(0xd800 | ((cp - 0x10000) >> 10));
//         result.emplace_back(0xdc00 | ((cp - 0x10000) & 0x03ff));
//         return result;
//     }
//     throw std::invalid_argument("failed to convert codepoint to utf16");
// }

// static std::vector<uint16_t> unicode_cpts_to_utf16(const
// std::vector<uint32_t> & cps) {
//     std::vector<uint16_t> result;
//     for (size_t i = 0; i < cps.size(); ++i) {
//         auto temp = unicode_cpt_to_utf16(cps[i]);
//         result.insert(result.end(), temp.begin(), temp.end());
//     }
//     return result;
// }

// static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16,
// size_t & offset) {
//     assert(offset < utf16.size());
//     if (((utf16[0] >> 10) << 10) != 0xd800) {
//         auto result = utf16[offset + 0];
//         offset += 1;
//         return result;
//     }
//
//     if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//         throw std::invalid_argument("invalid character");
//     }
//
//     auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] &
//     0x03ff)); offset += 2; return result;
// }

// static std::vector<uint32_t> unicode_cpts_from_utf16(const
// std::vector<uint16_t> & utf16) {
//     std::vector<uint32_t> result;
//     size_t offset = 0;
//     while (offset < utf16.size()) {
//         result.push_back(unicode_cpt_from_utf16(utf16, offset));
//     }
//     return result;
// }

static std::vector<codepoint_flags> unicode_cpt_flags_array() {
  std::vector<codepoint_flags> cpt_flags(MAX_CODEPOINTS, codepoint_flags::UNDEFINED);

  assert(unicode_ranges_flags.begin()[0].first == 0);
  assert(unicode_ranges_flags.begin()[unicode_ranges_flags.size() - 1].first == MAX_CODEPOINTS);
  for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
    const auto range_ini = unicode_ranges_flags.begin()[i - 1];  // codepoint_ini, flags
    const auto range_end = unicode_ranges_flags.begin()[i];      // codepoint_end, flags
    for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) { cpt_flags[cpt] = range_ini.second; }
  }

  for (auto cpt : unicode_set_whitespace) { cpt_flags[cpt].is_whitespace = true; }

  for (auto p : unicode_map_lowercase) { cpt_flags[p.second].is_lowercase = true; }

  for (auto p : unicode_map_uppercase) { cpt_flags[p.second].is_uppercase = true; }

  for (auto& range : unicode_ranges_nfd) {  // start, last, nfd
    cpt_flags[range.nfd].is_nfd = true;
  }

  return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
  std::unordered_map<uint8_t, std::string> map;
  for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
    assert(0 <= ch && ch < 256);
    map[ch] = unicode_cpt_to_utf8(ch);
  }
  for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
    assert(0 <= ch && ch < 256);
    map[ch] = unicode_cpt_to_utf8(ch);
  }
  for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
    assert(0 <= ch && ch < 256);
    map[ch] = unicode_cpt_to_utf8(ch);
  }
  auto n = 0;
  for (int ch = 0; ch < 256; ++ch) {
    if (map.find(ch) == map.end()) {
      map[ch] = unicode_cpt_to_utf8(256 + n);
      ++n;
    }
  }
  return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
  std::unordered_map<std::string, uint8_t> map;
  for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
    assert(0 <= ch && ch < 256);
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
    assert(0 <= ch && ch < 256);
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
    assert(0 <= ch && ch < 256);
    map[unicode_cpt_to_utf8(ch)] = ch;
  }
  auto n = 0;
  for (int ch = 0; ch < 256; ++ch) {
    if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
      map[unicode_cpt_to_utf8(256 + n)] = ch;
      ++n;
    }
  }
  return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string& s) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  return conv.from_bytes(s);
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string>& bpe_words) {
  std::vector<std::string> bpe_encoded_words;
  for (const auto& word : bpe_words) {
    std::string text_utf;
    auto utf_word = unicode_cpts_from_utf8(word);
    for (size_t i = 0; i < utf_word.size(); ++i) { text_utf += unicode_cpt_to_utf8(utf_word[i]); }

    std::string encoded_token;
    for (char& c : text_utf) { encoded_token += unicode_byte_to_utf8(c); }
    bpe_encoded_words.emplace_back(encoded_token);
  }
  return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string& text, const std::vector<size_t>& offsets) {
  std::vector<size_t> bpe_offsets;      // store the offset of each word
  bpe_offsets.reserve(offsets.size());  // Reserve memory for the approximate size

  const auto cpts = unicode_cpts_from_utf8(text);

  size_t start = 0;
  for (auto offset : offsets) {
    const size_t offset_ini = start;
    const size_t offset_end = start + offset;
    assert(offset_end <= cpts.size());
    start = offset_end;

    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&](const size_t pos) -> uint32_t {
      return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };

    auto _get_flags = [&](const size_t pos) -> codepoint_flags {
      return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos]) : codepoint_flags{};
    };

    size_t _prev_end = offset_ini;
    auto _add_token = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t len = end - _prev_end;
      if (len > 0) { bpe_offsets.push_back(len); }
      _prev_end = end;
      // if (len > 0) {
      //     std::string s = "";
      //     for(size_t p = end-len; p < end; p++)
      //         s += unicode_cpt_to_utf8(cpts[p]);
      //     printf(">>> '%s'\n", s.c_str());
      // }
      return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/) {
      const uint32_t cpt = _get_cpt(pos);
      const auto flags = _get_flags(pos);

      // regex: 's|'t|'re|'ve|'m|'ll|'d
      if (cpt == '\'' && pos + 1 < offset_end) {
        uint32_t cpt_next = _get_cpt(pos + 1);
        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
          pos += _add_token(pos + 2);
          continue;
        }
        if (pos + 2 < offset_end) {
          uint32_t cpt_next_next = _get_cpt(pos + 2);
          if ((cpt_next == 'r' && cpt_next_next == 'e') || (cpt_next == 'v' && cpt_next_next == 'e')
              || (cpt_next == 'l' && cpt_next_next == 'l')) {
            pos += _add_token(pos + 3);
            continue;
          }
        }
      }

      auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
      // regex: <space>?\p{L}+
      if (flags2.is_letter) {
        pos += (cpt == ' ');
        while (flags2.is_letter) { flags2 = _get_flags(++pos); }
        _add_token(pos);
        continue;
      }
      // regex: <space>?\p{N}+
      if (flags2.is_number) {
        pos += (cpt == ' ');
        while (flags2.is_number) { flags2 = _get_flags(++pos); }
        _add_token(pos);
        continue;
      }
      // regex: <space>?[^\s\p{L}\p{N}]+
      if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
        pos += (cpt == ' ');
        while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
          flags2 = _get_flags(++pos);
        }
        _add_token(pos);
        continue;
      }

      size_t num_whitespaces = 0;
      while (_get_flags(pos + num_whitespaces).is_whitespace) { num_whitespaces++; }

      // regex: \s+(?!\S)
      if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
        pos += num_whitespaces - 1;
        _add_token(pos);
        continue;
      }

      // regex: \s+
      if (num_whitespaces > 0) {
        pos += num_whitespaces;
        _add_token(pos);
        continue;
      }

      // no matches
      _add_token(++pos);
    }
  }

  return bpe_offsets;
}

// LLAMA3 system regex:
// "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string& text, const std::vector<size_t>& offsets) {
  std::vector<size_t> bpe_offsets;      // store the offset of each word
  bpe_offsets.reserve(offsets.size());  // Reserve memory for the approximate size

  const auto cpts = unicode_cpts_from_utf8(text);

  size_t start = 0;
  for (auto offset : offsets) {
    const size_t offset_ini = start;
    const size_t offset_end = start + offset;
    assert(offset_end <= cpts.size());
    start = offset_end;

    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&](const size_t pos) -> uint32_t {
      return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };

    auto _get_flags = [&](const size_t pos) -> codepoint_flags {
      return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos]) : codepoint_flags{};
    };

    size_t _prev_end = offset_ini;
    auto _add_token = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t len = end - _prev_end;
      if (len > 0) { bpe_offsets.push_back(len); }
      _prev_end = end;
      // if (len > 0) {
      //     std::string s = "";
      //     for(size_t p = end-len; p < end; p++)
      //         s += unicode_cpt_to_utf8(cpts[p]);
      //     printf(">>> '%s'\n", s.c_str());
      // }
      return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/) {
      const uint32_t cpt = _get_cpt(pos);
      const auto flags = _get_flags(pos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cpt == '\'' && pos + 1 < offset_end) {
        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
          pos += _add_token(pos + 2);
          continue;
        }
        if (pos + 2 < offset_end) {
          uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
          if ((cpt_next == 'r' && cpt_next_next == 'e') || (cpt_next == 'v' && cpt_next_next == 'e')
              || (cpt_next == 'l' && cpt_next_next == 'l')) {
            pos += _add_token(pos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
        if (flags.is_letter || _get_flags(pos + 1).is_letter) {  // one or more letters
          pos++;
          while (_get_flags(pos).is_letter) { pos++; }
          _add_token(pos);
          continue;
        }
      }

      // regex: \p{N}{1,3}
      if (flags.is_number) {
        size_t ini = pos;
        while (_get_flags(pos).is_number) {
          if (++pos - ini >= 3) {
            _add_token(pos);
            ini = pos;
          }
        }
        _add_token(pos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
      if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
        pos += (cpt == ' ');
        while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
          flags2 = _get_flags(++pos);
        }
        uint32_t cpt2 = _get_cpt(pos);
        while (cpt2 == '\r' || cpt2 == '\n') { cpt2 = _get_cpt(++pos); }
        _add_token(pos);
        continue;
      }

      size_t num_whitespaces = 0;
      size_t last_end_r_or_n = 0;
      while (_get_flags(pos + num_whitespaces).is_whitespace) {
        uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
        if (cpt2 == '\r' || cpt2 == '\n') { last_end_r_or_n = pos + num_whitespaces + 1; }
        num_whitespaces++;
      }

      // regex: \s*[\r\n]+
      if (last_end_r_or_n > 0) {
        pos = last_end_r_or_n;
        _add_token(pos);
        continue;
      }

      // regex: \s+(?!\S)
      if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
        pos += num_whitespaces - 1;
        _add_token(pos);
        continue;
      }

      // regex: \s+
      if (num_whitespaces > 0) {
        pos += num_whitespaces;
        _add_token(pos);
        continue;
      }

      // no matches
      _add_token(++pos);
    }
  }

  return bpe_offsets;
}

// use std::wregex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::wstring& wtext, const std::wstring& regex_expr,
                                                   const std::vector<size_t>& offsets) {
  std::wregex expr(regex_expr);
  std::vector<size_t> bpe_offsets;      // store the offset of each word
  bpe_offsets.reserve(offsets.size());  // Reserve memory for the approximate size
  size_t start = 0;
  for (auto offset : offsets) {
    std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
    std::wcregex_iterator end;

    int64_t start_idx = 0;
    while (it != end) {
      std::wcmatch match = *it;
      if (match.position() > start_idx) { bpe_offsets.emplace_back(match.position() - start_idx); }
      bpe_offsets.emplace_back(match.length());
      start_idx = match.position() + match.length();
      ++it;
    }

    if (start_idx < (int64_t)offset) { bpe_offsets.emplace_back(offset - start_idx); }
    start += offset;
  }

  return bpe_offsets;
}

// use std::regex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::string& text, const std::string& regex_expr,
                                                   const std::vector<size_t>& offsets) {
  std::regex expr(regex_expr);
  std::vector<size_t> bpe_offsets;      // store the offset of each word
  bpe_offsets.reserve(offsets.size());  // Reserve memory for the approximate size
  size_t start = 0;
  for (auto offset : offsets) {
    std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
    std::cregex_iterator end;

    int64_t start_idx = 0;
    while (it != end) {
      std::cmatch match = *it;
      if (match.position() > start_idx) { bpe_offsets.emplace_back(match.position() - start_idx); }
      bpe_offsets.emplace_back(match.length());
      start_idx = match.position() + match.length();
      ++it;
    }

    if (start_idx < (int64_t)offset) { bpe_offsets.emplace_back(offset - start_idx); }
    start += offset;
  }

  return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string& text, const std::string& regex_expr,
                                                      const std::vector<size_t>& offsets) {
  std::vector<size_t> bpe_offsets;

  if (regex_expr
      == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
         "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
    bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
  } else if (regex_expr
                 == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,"
                    "3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
             || regex_expr
                    == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^"
                       "\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
                       "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
  }

  return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cp) {
  std::string result;

  if (/* 0x00 <= cp && */ cp <= 0x7f) {
    result.push_back(cp);
    return result;
  }
  if (0x80 <= cp && cp <= 0x7ff) {
    result.push_back(0xc0 | ((cp >> 6) & 0x1f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x800 <= cp && cp <= 0xffff) {
    result.push_back(0xe0 | ((cp >> 12) & 0x0f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x10000 <= cp && cp <= 0x10ffff) {
    result.push_back(0xf0 | ((cp >> 18) & 0x07));
    result.push_back(0x80 | ((cp >> 12) & 0x3f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }

  throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t>& cpts) {
  auto comp = [](const uint32_t cpt, const range_nfd& range) { return cpt < range.first; };
  std::vector<uint32_t> result(cpts.size());
  for (size_t i = 0; i < cpts.size(); ++i) {
    const uint32_t cpt = cpts[i];
    auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
    result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
  }
  return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8) {
  std::vector<uint32_t> result;
  result.reserve(utf8.size());
  size_t offset = 0;
  while (offset < utf8.size()) { result.push_back(unicode_cpt_from_utf8(utf8, offset)); }
  return result;
}

codepoint_flags unicode_cpt_flags(const uint32_t cp) {
  static const codepoint_flags undef(codepoint_flags::UNDEFINED);
  static const auto cpt_flags = unicode_cpt_flags_array();
  return cp < cpt_flags.size() ? cpt_flags[cp] : undef;
}

codepoint_flags unicode_cpt_flags(const std::string& utf8) {
  static const codepoint_flags undef(codepoint_flags::UNDEFINED);
  if (utf8.empty()) {
    return undef;  // undefined
  }
  size_t offset = 0;
  return unicode_cpt_flags(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
  static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
  return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string& utf8) {
  static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
  return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cp) {
  // binary search
  auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cp,
                             [](const std::pair<uint32_t, uint32_t>& pair, uint32_t value) { return pair.first < value; });
  if (it != unicode_map_lowercase.end() && it->first == cp) { return it->second; }
  return cp;  // Return the original code point if no lowercase mapping is found
}

std::vector<std::string> unicode_regex_split(const std::string& text, const std::vector<std::string>& regex_exprs) {
  // unicode categories
  static const std::map<std::string, int> k_ucat_enum = {
      {"\\p{N}", codepoint_flags::NUMBER},
      {"\\p{L}", codepoint_flags::LETTER},
      {"\\p{P}", codepoint_flags::PUNCTUATION},
  };

  static const std::map<int, int> k_ucat_cpt = {
      {codepoint_flags::NUMBER, 0xD1},
      {codepoint_flags::LETTER, 0xD2},
      {codepoint_flags::PUNCTUATION, 0xD3},
  };

  static const std::map<int, std::string> k_ucat_map = {
      {codepoint_flags::NUMBER, "\x30-\x39"},           // 0-9
      {codepoint_flags::LETTER, "\x41-\x5A\x61-\x7A"},  // A-Za-z
      {codepoint_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-"
                                     "\\\x5D\x5F\\\x7B\\\x7D"},  // !-#%-*,-/:-;?-@\[-\]_\{\}
  };

  // compute collapsed codepoints only if needed by at least one regex
  bool need_collapse = false;
  for (auto& regex_expr : regex_exprs) {
    // search for unicode categories
    for (const auto& ucat : k_ucat_enum) {
      if (std::string::npos != regex_expr.find(ucat.first)) {
        need_collapse = true;
        break;
      }
    }
  }

  const auto cpts = unicode_cpts_from_utf8(text);

  // generate a "collapsed" representation of the text, where all codepoints are
  // replaced by a single byte ref:
  // https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2081479935
  std::string text_collapsed;
  if (need_collapse) {
    // collapse all unicode categories
    text_collapsed.resize(cpts.size());

    for (size_t i = 0; i < cpts.size(); ++i) {
      // keep single-byte codepoints as is
      if (cpts[i] < 128) {
        text_collapsed[i] = cpts[i];
        continue;
      }

      const auto flags = unicode_cpt_flags(cpts[i]);

      if (flags.is_whitespace) {
        // NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex
        // does. text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace
        // fallback
        text_collapsed[i] = (char)0x0B;  // <vertical tab> as whitespace fallback
      } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
        text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
      } else {
        text_collapsed[i] = (char)0xD0;  // fallback
      }
    }
  }

  std::vector<size_t> bpe_offsets = {cpts.size()};

  for (auto& regex_expr : regex_exprs) {
    // first, see if we have an efficient custom regex implementation
    auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

    if (!tmp.empty()) {
      bpe_offsets = std::move(tmp);
      continue;
    }

    // fallback to general-purpose std::regex / std::wregex
    try {
      // if a unicode category is used in the regex, we use the collapsed text
      // and replace the unicode category with the corresponding collapsed
      // representation
      bool use_collapsed = false;
      for (auto& ucat : k_ucat_enum) {
        if (std::string::npos != regex_expr.find(ucat.first)) {
          use_collapsed = true;
          break;
        }
      }

      if (use_collapsed) {
        // sanity-check that the original regex does not contain any non-ASCII
        // characters
        const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
        for (size_t i = 0; i < cpts_regex.size(); ++i) {
          if (cpts_regex[i] >= 128) {
            throw std::runtime_error("Regex includes both unicode categories and non-ASCII "
                                     "characters - not supported");
          }
        }

        // generate a collapsed representation of the regex
        std::string regex_expr_collapsed;

        // track if we are inside [], because nested [] are not allowed
        bool inside = false;
        for (size_t i = 0; i < regex_expr.size(); ++i) {
          if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
            regex_expr_collapsed += '[';
            inside = true;
            continue;
          }

          if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
            regex_expr_collapsed += ']';
            inside = false;
            continue;
          }

          if (regex_expr[i + 0] == '\\' && i + 4 < regex_expr.size() && regex_expr[i + 1] == 'p' && regex_expr[i + 2] == '{'
              && regex_expr[i + 4] == '}') {
            const std::string pat = regex_expr.substr(i, 5);
            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
              if (!inside) { regex_expr_collapsed += '['; }
              regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
              regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
              if (!inside) { regex_expr_collapsed += ']'; }
              i += 4;
              continue;
            }
          }

          regex_expr_collapsed += regex_expr[i];
        }

        // printf("text_collapsed: %s\n", text_collapsed.c_str());
        // printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
        bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
      } else {
        // no unicode category used, we can use std::wregex directly
        const std::wstring wregex_expr = unicode_wstring_from_utf8(regex_expr);

        // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as
        // fallback
        std::wstring wtext(cpts.begin(), cpts.end());
        for (size_t i = 0; i < wtext.size(); ++i) {
          if (wtext[i] > 0x7F && unicode_cpt_flags(wtext[i]).is_whitespace) { wtext[i] = 0x0B; }
        }

        // printf("text: %s\n", text.c_str());
        // printf("regex_expr: %s\n", regex_expr.c_str());
        bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
      }
    } catch (std::regex_error& e) {
      fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
      fprintf(stderr, "Regex error: %s\n", e.what());
      throw std::runtime_error("Failed to process regex");
    }
  }

  std::vector<std::string> bpe_words;
  bpe_words.reserve(bpe_offsets.size());  // reserve memory for the approximate size

  size_t start = 0;
  for (size_t& offset : bpe_offsets) {
    bpe_words.emplace_back();
    for (size_t i = start; i < start + offset; ++i) { bpe_words.back() += unicode_cpt_to_utf8(cpts[i]); }
    start += offset;
  }

  return unicode_byte_encoding_process(bpe_words);
}

// Get canonical combining class for a codepoint using existing flags data
static uint8_t get_combining_class(uint32_t cpt) {
  codepoint_flags flags = unicode_cpt_flags(cpt);

  // Use the existing flag system to determine combining class
  if (flags.is_accent_mark) {
    // Most combining marks have class 230, but some have different classes
    // This is a simplified mapping based on common Unicode patterns
    if (cpt >= 0x0591 && cpt <= 0x05BD) return 220;  // Hebrew accents
    if (cpt >= 0x05BF && cpt <= 0x05C7) return 230;  // Hebrew points
    if (cpt >= 0x0610 && cpt <= 0x061A) return 230;  // Arabic marks
    if (cpt >= 0x064B && cpt <= 0x065F) return 30;   // Arabic vowels
    if (cpt >= 0x0670 && cpt <= 0x0670) return 35;   // Arabic superscript alef
    if (cpt >= 0x06D6 && cpt <= 0x06E4) return 230;  // Arabic small high marks
    if (cpt >= 0x06E7 && cpt <= 0x06E8) return 230;  // Arabic small high marks
    if (cpt >= 0x06EA && cpt <= 0x06ED) return 220;  // Arabic small low marks

    // Default combining class for most combining marks
    return 230;
  }

  return 0;  // Non-combining character (starter)
}

// Apply canonical ordering using bubble sort (simple but correct)
static void canonical_order(std::vector<uint32_t>& cpts) {
  for (size_t i = 1; i < cpts.size(); ++i) {
    for (size_t j = i; j > 0; --j) {
      uint8_t cc1 = get_combining_class(cpts[j - 1]);
      uint8_t cc2 = get_combining_class(cpts[j]);

      // Only reorder if both have non-zero combining class and are out of order
      if (cc1 > cc2 && cc2 != 0) {
        std::swap(cpts[j - 1], cpts[j]);
      } else {
        break;
      }
    }
  }
}

// Build composition table by reverse-engineering the NFD data
static std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> build_composition_table() {
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> composition_map;

  // Iterate through all NFD mappings to build reverse composition table
  for (const auto& range : unicode_ranges_nfd) {
    for (uint32_t cpt = range.first; cpt <= range.last; ++cpt) {
      uint32_t base = range.nfd;

      // For NFC, we need to figure out what combining character was removed
      // This is a simplified approach that works for the most common cases

      // Common diacritic mappings based on the composed character
      uint32_t combining = 0;

      // Determine combining character based on the composed character
      // This is derived from common Unicode patterns
      switch (cpt) {
        // Grave accent (0x0300)
        case 0x00C0:
        case 0x00E0:  // À à
        case 0x00C8:
        case 0x00E8:  // È è
        case 0x00CC:
        case 0x00EC:  // Ì ì
        case 0x00D2:
        case 0x00F2:  // Ò ò
        case 0x00D9:
        case 0x00F9:  // Ù ù
        case 0x01CD:
        case 0x01CE:  // Ǎ ǎ
        case 0x01CF:
        case 0x01D0:  // Ǐ ǐ
        case 0x01D1:
        case 0x01D2:  // Ǒ ǒ
        case 0x01D3:
        case 0x01D4:  // Ǔ ǔ
          combining = 0x0300;
          break;

        // Acute accent (0x0301)
        case 0x00C1:
        case 0x00E1:  // Á á
        case 0x00C9:
        case 0x00E9:  // É é
        case 0x00CD:
        case 0x00ED:  // Í í
        case 0x00D3:
        case 0x00F3:  // Ó ó
        case 0x00DA:
        case 0x00FA:  // Ú ú
        case 0x00DD:
        case 0x00FD:  // Ý ý
          combining = 0x0301;
          break;

        // Circumflex (0x0302)
        case 0x00C2:
        case 0x00E2:  // Â â
        case 0x00CA:
        case 0x00EA:  // Ê ê
        case 0x00CE:
        case 0x00EE:  // Î î
        case 0x00D4:
        case 0x00F4:  // Ô ô
        case 0x00DB:
        case 0x00FB:  // Û û
          combining = 0x0302;
          break;

        // Tilde (0x0303)
        case 0x00C3:
        case 0x00E3:  // Ã ã
        case 0x00D1:
        case 0x00F1:  // Ñ ñ
        case 0x00D5:
        case 0x00F5:  // Õ õ
          combining = 0x0303;
          break;

        // Diaeresis (0x0308)
        case 0x00C4:
        case 0x00E4:  // Ä ä
        case 0x00CB:
        case 0x00EB:  // Ë ë
        case 0x00CF:
        case 0x00EF:  // Ï ï
        case 0x00D6:
        case 0x00F6:  // Ö ö
        case 0x00DC:
        case 0x00FC:  // Ü ü
        case 0x00FF:  // ÿ
          combining = 0x0308;
          break;

        // Ring above (0x030A)
        case 0x00C5:
        case 0x00E5:  // Å å
          combining = 0x030A;
          break;

        // Cedilla (0x0327)
        case 0x00C7:
        case 0x00E7:  // Ç ç
          combining = 0x0327;
          break;

        default:
          // For other characters, try to infer from Unicode blocks
          if (cpt >= 0x0100 && cpt <= 0x017F) {
            // Extended Latin A - try common patterns
            if ((cpt & 1) == 0) {  // Even codepoints (uppercase)
              if (cpt >= 0x0100 && cpt <= 0x0105)
                combining = 0x0304;  // macron
              else if (cpt >= 0x0102 && cpt <= 0x0107)
                combining = 0x0306;  // breve
              else if (cpt >= 0x0104 && cpt <= 0x0119)
                combining = 0x0328;  // ogonek
              else if (cpt >= 0x0106 && cpt <= 0x010D)
                combining = 0x0301;  // acute
              else if (cpt >= 0x0108 && cpt <= 0x010F)
                combining = 0x0302;  // circumflex
              else if (cpt >= 0x010A && cpt <= 0x0111)
                combining = 0x0307;  // dot above
              else if (cpt >= 0x010C && cpt <= 0x0165)
                combining = 0x030C;  // caron
            }
          }
          break;
      }

      // Only add to composition table if we identified a combining character
      if (combining != 0) { composition_map[{base, combining}] = cpt; }
    }
  }

  return composition_map;
}

// Get the composition table (built once, cached)
static const std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& get_composition_table() {
  static const auto composition_table = build_composition_table();
  return composition_table;
}

std::vector<uint32_t> unicode_cpts_normalize_nfc(const std::vector<uint32_t>& cpts) {
  // Step 1: Apply NFD (canonical decomposition) using existing implementation
  std::vector<uint32_t> nfd_result = unicode_cpts_normalize_nfd(cpts);

  // Step 2: Apply canonical ordering
  canonical_order(nfd_result);

  // Step 3: Apply canonical composition
  const auto& composition_table = get_composition_table();
  std::vector<uint32_t> result;
  result.reserve(nfd_result.size());

  size_t i = 0;
  while (i < nfd_result.size()) {
    uint32_t starter = nfd_result[i];
    result.push_back(starter);

    // Only try to compose if this is a starter (combining class 0)
    if (get_combining_class(starter) == 0) {
      size_t last_starter_pos = result.size() - 1;

      // Look for composable combining marks after this starter
      size_t j = i + 1;
      while (j < nfd_result.size()) {
        uint32_t combining = nfd_result[j];
        uint8_t cc = get_combining_class(combining);

        // If we hit another starter, stop
        if (cc == 0) break;

        // Try to compose with the last starter
        auto key = std::make_pair(result[last_starter_pos], combining);
        auto it = composition_table.find(key);

        if (it != composition_table.end()) {
          // Compose: replace starter with composed character
          result[last_starter_pos] = it->second;
          // Skip this combining character
          ++j;
          continue;
        }

        // No composition possible, add the combining character
        result.push_back(combining);
        ++j;
      }
      i = j;
    } else {
      ++i;
    }
  }

  return result;
}
