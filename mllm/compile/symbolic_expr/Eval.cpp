// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include <sstream>

#include "mllm/compile/symbolic_expr/Eval.hpp"
#include "mllm/compile/symbolic_expr/AST.hpp"
#include "mllm/compile/symbolic_expr/Parser.hpp"

namespace mllm {
SymExpr::SymExpr(const std::string& src) { parse(src); }

SymExpr::SymExpr(int32_t v) { ast_ = std::make_shared<IntLiteralAST>(v); }

SymExpr::SymExpr(float v) { ast_ = std::make_shared<FPLiteralAST>(v); }

SymExpr::SymExpr(const std::shared_ptr<SymbolExprAST>& ast) : ast_(ast) {}

SymExpr SymExpr::ceil() {
  auto new_ast = std::make_shared<UnaryExprAST>(SymbolExprTokenValue::Ceil, ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::floor() {
  auto new_ast = std::make_shared<UnaryExprAST>(SymbolExprTokenValue::Floor, ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::abs() {
  auto new_ast = std::make_shared<UnaryExprAST>(SymbolExprTokenValue::Abs, ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::operator+(const SymExpr& rhs) const {
  auto new_ast = std::make_shared<BinaryExprAST>(SymbolExprTokenValue::Plus, ast_, rhs.ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::operator-(const SymExpr& rhs) const {
  auto new_ast = std::make_shared<BinaryExprAST>(SymbolExprTokenValue::Minus, ast_, rhs.ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::operator*(const SymExpr& rhs) const {
  auto new_ast = std::make_shared<BinaryExprAST>(SymbolExprTokenValue::Multiply, ast_, rhs.ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::operator/(const SymExpr& rhs) const {
  auto new_ast = std::make_shared<BinaryExprAST>(SymbolExprTokenValue::Divide, ast_, rhs.ast_);
  return SymExpr(new_ast);
}

SymExpr SymExpr::operator%(const SymExpr& rhs) const {
  auto new_ast = std::make_shared<BinaryExprAST>(SymbolExprTokenValue::Mod, ast_, rhs.ast_);
  return SymExpr(new_ast);
}

bool SymExpr::isStatic() const {
  bool flag = true;
  ast_->isStatic(flag);
  return flag;
}

float SymExpr::eval(const SymExprDict& dict) { return ast_->eval(dict); }

int32_t SymExpr::evalAsInt(const SymExprDict& dict) { return ast_->evalAsInt(dict); }

std::string SymExpr::str() const {
  std::stringstream ss;
  ast_->str(ss);
  return ss.str();
}

void SymExpr::parse(const std::string& src) {
  auto lexer = SymbolExprLexer(src);
  auto parser = SymbolExprParser(&lexer);
  ast_ = parser.parse();
}
}  // namespace mllm
