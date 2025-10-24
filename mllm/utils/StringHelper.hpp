// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <cctype>

namespace mllm {
inline std::vector<std::string> splitString(const std::string& s, const std::string& sep = "", int maxsplit = -1) {
  std::vector<std::string> out;
  if (maxsplit == 0) {
    out.push_back(s);
    return out;
  }

  const char* p = s.data();
  const char* end = p + s.size();

  if (sep.empty()) {
    auto skip_space = [&]() {
      while (p != end && std::isspace(static_cast<unsigned char>(*p))) ++p;
    };
    skip_space();
    while (p != end) {
      const char* start = p;
      while (p != end && !std::isspace(static_cast<unsigned char>(*p))) ++p;
      out.emplace_back(start, p);
      if (maxsplit >= 0 && --maxsplit == 0) {
        out.emplace_back(p, end);
        return out;
      }
      skip_space();
    }
    return out;
  }

  if (sep.size() == 1) {
    const unsigned char needle = static_cast<unsigned char>(sep[0]);
    while (maxsplit != 0) {
      const char* pos = reinterpret_cast<const char*>(std::memchr(p, needle, end - p));
      if (!pos) break;
      out.emplace_back(p, pos);
      p = pos + 1;
      if (maxsplit > 0) --maxsplit;
    }
  } else {
    const auto n = sep.size();
    while (maxsplit != 0) {
      const char* pos = std::search(p, end, sep.begin(), sep.end());
      if (pos == end) break;
      out.emplace_back(p, pos);
      p = pos + n;
      if (maxsplit > 0) --maxsplit;
    }
  }
  out.emplace_back(p, end);
  return out;
}
}  // namespace mllm
