// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>

#include "mllm/compile/symbolic_expr/AST.hpp"
#include "mllm/compile/symbolic_expr/Tokens.hpp"

namespace mllm {

struct SymbolExprToken {
  SymbolExprToken() : type_(SymbolExprTokenType::None), token_v_(SymbolExprTokenValue::Unused), int_v_(0), fp_v_(0.f) {}

  explicit SymbolExprToken(const SymbolExprTokenType type)
      : type_(type), token_v_(SymbolExprTokenValue::None), int_v_(0), fp_v_(0.f) {}

  SymbolExprToken(const SymbolExprTokenType type, const SymbolExprTokenValue token_v)
      : type_(type), token_v_(token_v), int_v_(0), fp_v_(0.f) {}

  SymbolExprToken(const SymbolExprTokenType type, const SymbolExprTokenValue token_v, int32_t int_v)
      : type_(type), token_v_(token_v), int_v_(int_v), fp_v_(0.f) {}

  SymbolExprToken(const SymbolExprTokenType type, const SymbolExprTokenValue token_v, float fp_v)
      : type_(type), token_v_(token_v), int_v_(0), fp_v_(fp_v) {}

  SymbolExprToken(const SymbolExprTokenType type, const SymbolExprTokenValue token_v, std::string str_v)
      : type_(type), token_v_(token_v), int_v_(0), fp_v_(0.f), str_v_(std::move(str_v)) {}

  SymbolExprTokenType type_;
  SymbolExprTokenValue token_v_;
  int32_t int_v_;
  float fp_v_;
  std::string str_v_;
  std::string name_;
};

class SymbolExprTokenDic {
  using token_meta_t = std::tuple<SymbolExprTokenType, SymbolExprTokenValue, int16_t>;

 public:
  SymbolExprTokenDic();

  [[nodiscard]] token_meta_t lookup(const std::string& name) const;

  [[nodiscard]] bool haveToken(const std::string& name) const;

 private:
  void addToken(const std::string& name, const token_meta_t& token_meta);

  // key: token name
  // value: (token-type, token-value, precedence)
  std::unordered_map<std::string, token_meta_t> dictionary_;
};

class SymbolExprLexer {
  enum LexerState {
    kNone = 0,
    kEof,
    kIdentifier,
    kNumber,
    kOperations,
  };

 public:
  explicit SymbolExprLexer(const std::string& src) : ss_(src) {}

  void resetWithNewString(const std::string& src);

  SymbolExprToken getToken() { return cur_token_; }

  SymbolExprToken getNextToken();

  int32_t getPrecedence(const SymbolExprToken& t);

 private:
  void handleInt();

  void handleFP();

  void handleNumberState();

  char peekChar();

  void getNextChar();

  void handleEOFState();

  void handleIdentifierState();

  void handleOperationState();

  void addToCharBuffer(char c);

  void clearCharBuffer();

 private:
  SymbolExprToken cur_token_;
  LexerState state_ = kNone;
  char cur_char_ = 0;
  std::stringstream ss_;
  SymbolExprTokenDic token_dictionary_;
  std::string char_buffer_;
};

class SymbolExprParser {
  using expr_ast_ptr_t = std::shared_ptr<SymbolExprAST>;

 public:
  explicit SymbolExprParser(SymbolExprLexer* lexer) : lexer_(lexer) {}

  expr_ast_ptr_t parse();

 private:
  expr_ast_ptr_t parseExpr(expr_ast_ptr_t& prev_expr);

  expr_ast_ptr_t parseGroupedExpr();

  expr_ast_ptr_t parseVarExpr();

  expr_ast_ptr_t parseBinaryExpr(expr_ast_ptr_t& prev_expr);

  expr_ast_ptr_t parseUnaryExpr();

  expr_ast_ptr_t parseNumberLiteralExpr();

  SymbolExprToken lookOneMoreToken();

  SymbolExprToken getToken();

  SymbolExprToken getNextToken();

 private:
  // token buffer for look ahead, just look 1 more token is enough. Like LL(1)
  std::queue<SymbolExprToken> token_buffer_;
  SymbolExprToken cur_token_;
  SymbolExprLexer* lexer_ = nullptr;
};

}  // namespace mllm
