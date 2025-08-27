// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cctype>
#include <cstdlib>
#include <memory>

#include "mllm/compile/symbolic_expr/Parser.hpp"
#include "mllm/compile/symbolic_expr/AST.hpp"

namespace mllm {

SymbolExprTokenDic::SymbolExprTokenDic() {
  // add operators
  ADD_TOKEN("+", Operators, Plus, 10);
  ADD_TOKEN("-", Operators, Minus, 10);
  ADD_TOKEN("%", Operators, Mod, 10);
  ADD_TOKEN("*", Operators, Multiply, 20);
  ADD_TOKEN("/", Operators, Divide, 20);

  // add keywords
  ADD_TOKEN("ceil", Keywords, Ceil, -1);
  ADD_TOKEN("floor", Keywords, Floor, -1);
  ADD_TOKEN("abs", Keywords, Abs, -1);

  // parentheses
  ADD_TOKEN("(", Delimiter, ParenthesesOpen, -1);
  ADD_TOKEN(")", Delimiter, ParenthesesClose, -1);
}

void SymbolExprTokenDic::addToken(const std::string& name, const token_meta_t& token_meta) {
  dictionary_.insert({name, token_meta});
}

SymbolExprTokenDic::token_meta_t SymbolExprTokenDic::lookup(const std::string& name) const {
  auto iter = dictionary_.find(name);
  if (iter == dictionary_.end()) {
    // the unused value must be Identifier.
    return token_meta_t({SymbolExprTokenType::Identifier, SymbolExprTokenValue::Unused, -1});
  } else {
    // Other wise is built in types.
    return iter->second;
  }
}

bool SymbolExprTokenDic::haveToken(const std::string& name) const { return dictionary_.find(name) != dictionary_.end(); }

void SymbolExprLexer::resetWithNewString(const std::string& src) {
  ss_.clear();
  ss_ = std::stringstream(src);
  state_ = kNone;
  cur_char_ = 0;
  char_buffer_.clear();
}

SymbolExprToken SymbolExprLexer::getNextToken() {
  bool matched = false;
  do {
    if (state_ != kNone) matched = true;

    switch (state_) {
      case kEof:
        handleEOFState();
        state_ = kNone;
        break;
      case kIdentifier:
        handleIdentifierState();
        state_ = kNone;
        break;
      case kNumber:
        handleNumberState();
        state_ = kNone;
        break;
      case kOperations:
        handleOperationState();
        state_ = kNone;
        break;
      case kNone:
        getNextChar();
        state_ = kNone;
        break;
    }

    if (state_ == kNone) {
      if (ss_.eof()) {
        state_ = kEof;
      } else {
        if (std::isalpha(cur_char_) || (cur_char_ == '_')) {
          // handle both keywords and var
          // keywords: ceil, floor, abs
          // var: "a", "b", "c"
          state_ = kIdentifier;
        } else if (std::isdigit(cur_char_) || (cur_char_ == '.')) {
          // handle all numbers
          // include integer and float.
          state_ = kNumber;
        } else if (cur_char_ == ' ') {
          // eats blank
          state_ = kNone;
        } else {
          // handle operations, such as
          // +,-,*,/
          //
          // NOTE:
          // Delimiter '(' and ')' also processed in the operation states, but
          // they will be marked delimiter in token's type scope.
          state_ = kOperations;
        }
      }
    }

  } while (!matched);

  return cur_token_;
}

int32_t SymbolExprLexer::getPrecedence(const SymbolExprToken& t) {
  auto [_1, _2, prec] = token_dictionary_.lookup(t.name_);
  return prec;
}

void SymbolExprLexer::handleInt() {
  // exponent is not supported.
  // int: 12, 1, 20, etc
  int32_t v = atoi(char_buffer_.c_str());
  cur_token_ = {SymbolExprTokenType::Int, SymbolExprTokenValue::I32, v};
  cur_token_.name_ = char_buffer_;
}

void SymbolExprLexer::handleFP() {
  // exponent is not supported.
  // float: 0.0, 0.12, 12.2, .12, etc
  float v = static_cast<float>(atof(char_buffer_.c_str()));
  cur_token_ = {SymbolExprTokenType::FP, SymbolExprTokenValue::F32, v};
  cur_token_.name_ = char_buffer_;
}

void SymbolExprLexer::handleNumberState() {
  // exponent is not supported.
  // int: 12, 1, 20, etc
  // float: 0.0, 0.12, 12.2, .12, etc
  bool is_int = true;
  bool is_float = false;
  bool has_one_dot = false;

  // check current char is "."
  if (cur_char_ == '.') {
    is_float = true;
    is_int = false;
    has_one_dot = true;
  }

  addToCharBuffer(cur_char_);
  getNextChar();

  auto isOk = [&]() -> bool {
    if (has_one_dot && cur_char_ == '.') {
      // find multiple dot.
      // FIXME: log an error
      exit(-1);
    }
    if (!has_one_dot && cur_char_ == '.') {
      is_float = true;
      is_int = false;
      return true;
    }
    if (std::isdigit(cur_char_)) { return true; }
    return false;
  };

  while (isOk()) {
    addToCharBuffer(cur_char_);
    getNextChar();
  }

  if (is_int) handleInt();
  if (is_float) handleFP();

  clearCharBuffer();
}

char SymbolExprLexer::peekChar() {
  // read the next char. but keep this char in the stream
  return static_cast<char>(ss_.peek());
}

void SymbolExprLexer::getNextChar() {
  // read the next char and pop this from the stream
  cur_char_ = static_cast<char>(ss_.get());
}

void SymbolExprLexer::handleEOFState() {
  cur_token_ = {SymbolExprTokenType::Eof, SymbolExprTokenValue::None};
  ss_.clear();
  state_ = kNone;
}

void SymbolExprLexer::handleIdentifierState() {
  // add current char to buffer
  addToCharBuffer(cur_char_);

  // get nextchar
  getNextChar();

  while (std::isalpha(cur_char_) || std::isdigit(cur_char_) || cur_char_ == '_') {
    addToCharBuffer(cur_char_);
    getNextChar();
  }

  if (token_dictionary_.haveToken(char_buffer_)) {
    // the token is keywords
    // such as floor, ceil, abs
    auto [token_type, token_value, prec] = token_dictionary_.lookup(char_buffer_);
    cur_token_ = {token_type, token_value};
    cur_token_.name_ = char_buffer_;
  } else {
    // the token is an var
    // such as "a", "b", "c"
    cur_token_ = {SymbolExprTokenType::Identifier, SymbolExprTokenValue::Unused};
    cur_token_.str_v_ = char_buffer_;
    cur_token_.name_ = char_buffer_;
  }
  clearCharBuffer();
}

void SymbolExprLexer::handleOperationState() {
  // assumed that all operations has only one char.
  addToCharBuffer(cur_char_);

  auto [token_type, token_value, prec] = token_dictionary_.lookup(char_buffer_);

  cur_token_ = {token_type, token_value};
  cur_token_.name_ = char_buffer_;

  clearCharBuffer();
  getNextChar();
}

void SymbolExprLexer::addToCharBuffer(char c) { char_buffer_.push_back(c); }

void SymbolExprLexer::clearCharBuffer() { char_buffer_.clear(); }

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parse() {
  auto token = getNextToken();
  expr_ast_ptr_t ret = nullptr;
  while (token.type_ != SymbolExprTokenType::Eof) {
    ret = parseExpr(ret);
    token = getNextToken();
  }
  return ret;
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseExpr(expr_ast_ptr_t& prev_expr) {
  expr_ast_ptr_t ret = nullptr;
  auto token = getToken();
  switch (token.type_) {
    case SymbolExprTokenType::Int:
    case SymbolExprTokenType::FP: ret = parseNumberLiteralExpr(); break;
    case SymbolExprTokenType::Keywords:
      // FIXME: currently, all keywords is unary operator. fix it to support multi
      // inputs with args and rets.
      ret = parseUnaryExpr();
      break;
    case SymbolExprTokenType::Delimiter: ret = parseGroupedExpr(); break;
    case SymbolExprTokenType::Operators:
      // FIXME: currently, all operators is binary operator
      ret = parseBinaryExpr(prev_expr);
      break;
    case SymbolExprTokenType::Identifier: ret = parseVarExpr(); break;
    default: break;
  }
  return ret;
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseGroupedExpr() {
  auto token = getNextToken();
  expr_ast_ptr_t ret = nullptr;
  while (token.token_v_ != SymbolExprTokenValue::ParenthesesClose) {
    ret = parseExpr(ret);
    token = getNextToken();
  }

  // ')' has already consumed by while loop.

  return ret;
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseVarExpr() {
  auto token = getToken();
  return std::make_shared<VarAST>(token.str_v_);
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseBinaryExpr(expr_ast_ptr_t& prev_expr) {
  auto binary_op = getToken();

  (void)getNextToken();
  auto rhs_expr = parseExpr(prev_expr);

  // look ahead, LL(1)
  auto next_op = lookOneMoreToken();
  if (next_op.type_ == SymbolExprTokenType::Operators && lexer_->getPrecedence(next_op) > lexer_->getPrecedence(binary_op)) {
    (void)getNextToken();
    rhs_expr = parseBinaryExpr(rhs_expr);
  }

  return std::make_shared<BinaryExprAST>(binary_op.token_v_, prev_expr, rhs_expr);
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseUnaryExpr() {
  auto unary_op = getToken();
  // the next token after a keyword must be a '('
  (void)getNextToken();

  auto group = parseGroupedExpr();

  return std::make_shared<UnaryExprAST>(unary_op.token_v_, group);
}

SymbolExprParser::expr_ast_ptr_t SymbolExprParser::parseNumberLiteralExpr() {
  auto token = getToken();
  if (token.type_ == SymbolExprTokenType::Int) {
    return std::make_shared<IntLiteralAST>(token.int_v_);
  } else if (token.type_ == SymbolExprTokenType::FP) {
    return std::make_shared<FPLiteralAST>(token.fp_v_);
  }
  return nullptr;
}

SymbolExprToken SymbolExprParser::lookOneMoreToken() {
  auto ret = lexer_->getNextToken();
  token_buffer_.push(ret);
  return ret;
}

SymbolExprToken SymbolExprParser::getToken() { return cur_token_; }

SymbolExprToken SymbolExprParser::getNextToken() {
  if (!token_buffer_.empty()) {
    auto ret = token_buffer_.front();
    token_buffer_.pop();
    cur_token_ = ret;
    return ret;
  }
  auto ret = lexer_->getNextToken();
  cur_token_ = ret;
  return ret;
}

}  // namespace mllm
