// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <vector>
#include <algorithm>

#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::preprocessor {

std::string wideString2Utf8String(const std::wstring& wstr) {
  std::string result;
  for (wchar_t wc : wstr) {
    if (wc <= 0x7FU) {
      result.push_back(static_cast<char>(wc));
    } else if (wc <= 0x7FFU) {
      result.push_back(static_cast<char>(0xC0U | ((wc >> 6U) & 0x1FU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
    } else if (wc <= 0xFFFFU) {
      result.push_back(static_cast<char>(0xE0U | ((wc >> 12U) & 0x0FU)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3FU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
    } else if (wc <= 0x10FFFFU) {
      result.push_back(static_cast<char>(0xF0U | ((wc >> 18U) & 0x07U)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 12U) & 0x3FU)));
      result.push_back(static_cast<char>(0x80U | ((wc >> 6U) & 0x3FU)));
      result.push_back(static_cast<char>(0x80U | (wc & 0x3FU)));
    }
  }
  return result;
}

std::wstring utf8string2WideString(const std::string& str) {
  std::wstring w_ret_string;
  for (unsigned int i = 0; i < str.size();) {
    auto byte = static_cast<unsigned char>(str[i]);
    if ((byte & 0x80U) == 0) {
      // 1-byte character
      w_ret_string.push_back(static_cast<wchar_t>(byte));
      ++i;
    } else if ((byte & 0xE0U) == 0xC0) {
      // 2-byte character
      if (i + 1 < str.size()) {
        wchar_t wc = (static_cast<wchar_t>(byte & 0x1FU) << 6U) | (static_cast<wchar_t>(str[i + 1] & 0x3FU));
        w_ret_string.push_back(wc);
        i += 2;
      } else {
        break;
      }
    } else if ((byte & 0xF0U) == 0xE0U) {
      // 3-byte character
      if (i + 2 < str.size()) {
        wchar_t wc = (static_cast<wchar_t>(byte & 0x0FU) << 12U) | (static_cast<wchar_t>(str[i + 1] & 0x3FU) << 6U)
                     | (static_cast<wchar_t>(str[i + 2] & 0x3FU));
        w_ret_string.push_back(wc);
        i += 3;
      } else {
        break;
      }
    } else if ((byte & 0xF8U) == 0xF0U) {
      // 4-byte character
      if (i + 3 < str.size()) {
        wchar_t wc = (static_cast<wchar_t>(byte & 0x07U) << 18U) | (static_cast<wchar_t>(str[i + 1] & 0x3FU) << 12U)
                     | (static_cast<wchar_t>(str[i + 2] & 0x3FU) << 6U) | (static_cast<wchar_t>(str[i + 3] & 0x3FU));
        w_ret_string.push_back(wc);
        i += 4;
      } else {
        break;
      }
    } else {
      // Invalid UTF-8 sequence
      ++i;
    }
  }
  return w_ret_string;
}

void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t>& dict) {
  std::vector<std::wint_t> bs((ord(L"~") - ord(L"!") + 1) + (ord(L"¬") - ord(L"¡") + 1) + (ord(L"ÿ") - ord(L"®") + 1));

  int cnt = 0;
  for (std::wint_t i = ord(L"!"); i <= ord(L"~"); ++i) { bs[cnt++] = i; }
  for (std::wint_t i = ord(L"¡"); i <= ord(L"¬"); ++i) { bs[cnt++] = i; }
  for (std::wint_t i = ord(L"®"); i <= ord(L"ÿ"); ++i) { bs[cnt++] = i; }

  std::vector<std::wint_t> cs(bs.size());
  for (int i = 0; i < bs.size(); ++i) { cs[i] = bs[i]; }

  int n = 0;
  for (std::wint_t b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.emplace_back(b);
      cs.emplace_back(256 + n);
      ++n;
    }
  }

  std::vector<wchar_t> cs_chars(cs.size());
  for (int i = 0; i < cs.size(); ++i) { cs_chars[i] = chr(cs[i]); }
  for (int i = 0; i < bs.size(); ++i) { dict.insert({bs[i], cs_chars[i]}); }
}

}  // namespace mllm::preprocessor