// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// Platform-specific macro definitions
#ifdef _MSC_VER
// Use MSVC specific predefined macro for function signature
#define MLLM_PRETTY_FUNCTION __FUNCSIG__
// Helper macro to force expansion of __VA_ARGS__ in MSVC
#define MLLM_EXPAND(x) x
#else
// Assume GCC/Clang if not MSVC
#define MLLM_PRETTY_FUNCTION __PRETTY_FUNCTION__
#define MLLM_EXPAND(x) x  // No-op for GCC/Clang
#endif

#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <cctype>
#include <utility>

namespace mllm {

#define MLLM_DBG_PICK_IMPL_2_OR_0(dummy, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define MLLM_DBG_IMPL_TYPE_(...) \
  MLLM_EXPAND(MLLM_DBG_PICK_IMPL_2_OR_0(dummy, ##__VA_ARGS__, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0))

static inline std::vector<std::string> _mllm_dbg_split_args_name(const char* args_name) {
  std::vector<std::string> names;
  std::string input(args_name);
  if (input.empty()) { return names; }
  size_t start = 0;
  int paren_level = 0;

  for (size_t i = 0; i < input.length(); ++i) {
    if (input[i] == '(') {
      paren_level++;
    } else if (input[i] == ')') {
      paren_level--;
    } else if (input[i] == ',' && paren_level == 0) {
      std::string token = input.substr(start, i - start);
      token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
      token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                  token.end());
      names.push_back(token);
      start = i + 1;
    }
  }

  std::string token = input.substr(start);
  token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              token.end());
  if (!token.empty()) { names.push_back(token); }

  return names;
}

template<typename Tuple, size_t... Is>
static inline void _mllm_dbg_print_args_impl(const std::vector<std::string>& names, const Tuple& args_tuple,
                                             std::index_sequence<Is...>) {
  // C++17 fold expression to print all arguments.
  ((fmt::print("{}{}: {}", (Is == 0 ? "" : ", "), names[Is], std::get<Is>(args_tuple))), ...);
}

template<typename... Args>
static inline void _mllm_dbg_print_args(const char* args_name, Args... args) {
  auto names = _mllm_dbg_split_args_name(args_name);
  auto args_tuple = std::make_tuple(args...);
  _mllm_dbg_print_args_impl(names, args_tuple, std::index_sequence_for<Args...>{});
}

// Helper macros to concatenate tokens and create the final macro call.
#define __DGB_STR_CONCAT_IMPL(a, b) a##b
#define __DGB_STR_CONCAT(a, b) __DGB_STR_CONCAT_IMPL(a, b)

#define Dbg(...) MLLM_EXPAND(__DGB_STR_CONCAT(Dbg_IMPL_, MLLM_DBG_IMPL_TYPE_(__VA_ARGS__))(__VA_ARGS__))

// Implementation for Dbg() with no arguments.
#define Dbg_IMPL_0(...)                                                                          \
  fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, "dbg| {}:{} in ", __FILE__, __LINE__); \
  fmt::print("\"{}\"\n", MLLM_PRETTY_FUNCTION)

// Implementation for Dbg(...) with one or more arguments.
#define Dbg_IMPL_2(...)                                                                       \
  fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, "dbg| {}:{} ", __FILE__, __LINE__); \
  mllm::_mllm_dbg_print_args(#__VA_ARGS__, __VA_ARGS__);                                      \
  fmt::print(" in \"{}\"\n", MLLM_PRETTY_FUNCTION)

}  // namespace mllm
