// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace mllm {

#define ADD_TOKEN(name, type, value, precedence) \
  addToken(name, {SymbolExprTokenType::type, SymbolExprTokenValue::value, precedence})

enum class SymbolExprTokenType : int16_t {
  Int = 0,     // integer
  FP,          // floating point
  Operators,   // +, -, *, /, %, etc
  Keywords,    // floor, ceil, etc,
  Identifier,  // such as "H" in "H * W"
  Delimiter,
  Eof,
  None = -42,
};

enum class SymbolExprTokenValue : int16_t {
  // None
  None = 0,

  // builtin types
  I32 = -1,
  F32 = -2,

  // key words
  Ceil = -20,
  Floor = -21,
  Abs = -22,

  // operators
  Plus = '+',
  Minus = '-',
  Multiply = '*',
  Divide = '/',
  Mod = '%',

  // parentheses
  ParenthesesOpen = '(',
  ParenthesesClose = ')',

  // unused
  Unused = -42,
};

}  // namespace mllm
