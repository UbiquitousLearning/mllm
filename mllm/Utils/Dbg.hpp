/**
 * @file Dbg.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Do not use std::cout or printf or fmt::print when debuging any more!!!
 * @version 0.1
 * @date 2025-07-19
 *
 */
#pragma once

#ifdef _MSC_VER
#error "The Mllm's Dbg helper is built on GCC/Clang extension, which is not compatible with MSVC compiler."
#endif

#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>
#include <string>
#include <tuple>

namespace mllm {

#define MLLM_DBG_IMPL_TYPE_(...)                                                                                               \
  __MLLM_DBG_IMPL_TYPE_PRIVATE(0, ##__VA_ARGS__, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                               2, 2, 2, 2, 2, 2, 0)

#define __MLLM_DBG_IMPL_TYPE_PRIVATE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                                     _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, \
                                     _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, \
                                     _56, _57, _58, _59, _60, _61, _62, _63, _64, N, ...)                                      \
  N

static inline std::vector<std::string> _mllm_dbg_split_args_name(const char* args_name) {
  std::vector<std::string> names;
  std::string input(args_name);
  size_t start = 0;
  size_t end = 0;

  while ((end = input.find(',', start)) != std::string::npos) {
    std::string token = input.substr(start, end - start);
    token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                token.end());

    names.push_back(token);
    start = end + 1;
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
  ((fmt::print("{}{}:{}", (Is == 0 ? "" : ", "), names[Is], std::get<Is>(args_tuple))), ...);
}

template<typename... Args>
static inline void _mllm_dbg_print_args(const char* args_name, Args... args) {
  auto names = _mllm_dbg_split_args_name(args_name);
  auto args_tuple = std::make_tuple(args...);
  _mllm_dbg_print_args_impl(names, args_tuple, std::index_sequence_for<Args...>{});
}

#define __DGB_STR_CONCAT_IMPL(a, b) a##b
#define __DGB_STR_CONCAT(a, b) __DGB_STR_CONCAT_IMPL(a, b)

#define Dbg(...) __DGB_STR_CONCAT(Dbg_IMPL_, MLLM_DBG_IMPL_TYPE_(__VA_ARGS__))(__VA_ARGS__)

#define Dbg_IMPL_0(...)                                                                           \
  fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, "dbg| {}:{}, in ", __FILE__, __LINE__); \
  fmt::print("\"{}\"\n", __PRETTY_FUNCTION__)

#define Dbg_IMPL_2(...)                                                                                             \
  fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, "dbg| {}:{}, ", __FILE__, __LINE__, __PRETTY_FUNCTION__); \
  mllm::_mllm_dbg_print_args(#__VA_ARGS__, __VA_ARGS__);                                                            \
  fmt::print("\n")

}  // namespace mllm
