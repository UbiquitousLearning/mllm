// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>

#include <fmt/core.h>

#include "mllm/compile/symbolic_expr/Tokens.hpp"

namespace mllm {

class SymbolExprAST {
 public:
  enum Kinds {
    // un resolved
    Unresolved = 0,

    // binary op
    Binary,  // such as +, -, *, /
    Unary,   // such as floor, abs, ceil

    // var placeholder
    Var,

    // Literal
    IntLiteral,
    FPLiteral,
  };

  virtual ~SymbolExprAST() = default;

  explicit SymbolExprAST(Kinds kind) : kind_(kind) {}

  [[nodiscard]] Kinds getKind() const { return kind_; }

  // calculate using fp32
  float value_ = 0;

  virtual void dump(int32_t indent) {}

  virtual float eval(const std::unordered_map<std::string, float>& var_values) { return value_; }

  virtual void isStatic(bool& flag) {}

  virtual void str(std::stringstream& ss) {}

  int32_t evalAsInt(const std::unordered_map<std::string, float>& var_values) { return static_cast<int32_t>(eval(var_values)); }

 protected:
  void dumpIndent(int32_t indent) {
    for (auto i = 0; i < indent; ++i) fmt::print("  ");
  }

 private:
  const Kinds kind_ = Unresolved;
};

class BinaryExprAST final : public SymbolExprAST {
 public:
  BinaryExprAST(SymbolExprTokenValue op_type, const std::shared_ptr<SymbolExprAST>& lhs,
                const std::shared_ptr<SymbolExprAST>& rhs)
      : SymbolExprAST(Binary), op_type_(op_type), lhs_(lhs), rhs_(rhs) {}

  [[nodiscard]] SymbolExprTokenValue getOpType() const { return op_type_; }

  SymbolExprAST* getLhs() { return lhs_.get(); }

  SymbolExprAST* getRhs() { return rhs_.get(); }

  // LLVM style RTTI
  static bool classof(const SymbolExprAST* ast) { return Binary == ast->getKind(); }

  void dump(int32_t indent) override {
    // dump self
    dumpIndent(indent);
    fmt::print("binary {} op:\n", (char)op_type_);
    indent++;

    // dump childs
    lhs_->dump(indent);
    rhs_->dump(indent);
    indent--;
  }

  void isStatic(bool& flag) override {
    lhs_->isStatic(flag);
    rhs_->isStatic(flag);
  }

  float eval(const std::unordered_map<std::string, float>& var_values) override {
    auto lhs_v = lhs_->eval(var_values);
    auto rhs_v = rhs_->eval(var_values);
    switch (getOpType()) {
      case SymbolExprTokenValue::Plus: value_ = lhs_v + rhs_v; break;
      case SymbolExprTokenValue::Minus: value_ = lhs_v - rhs_v; break;
      case SymbolExprTokenValue::Multiply: value_ = lhs_v * rhs_v; break;
      case SymbolExprTokenValue::Divide: value_ = lhs_v / rhs_v; break;
      case SymbolExprTokenValue::Mod:
        value_ = static_cast<float>(static_cast<int32_t>(lhs_v) % static_cast<int32_t>(rhs_v));
        break;
      default: break;
    }
    return value_;
  }

  void str(std::stringstream& ss) override {
    if (lhs_->getKind() == Kinds::Binary) {
      ss << "(";
      lhs_->str(ss);
      ss << ")";
    } else {
      lhs_->str(ss);
    }
    switch (getOpType()) {
      case SymbolExprTokenValue::Plus: ss << " + "; break;
      case SymbolExprTokenValue::Minus: ss << " - "; break;
      case SymbolExprTokenValue::Multiply: ss << " * "; break;
      case SymbolExprTokenValue::Divide: ss << " / "; break;
      case SymbolExprTokenValue::Mod: ss << " % "; break;
      default: break;
    }
    if (rhs_->getKind() == Kinds::Binary) {
      ss << "(";
      rhs_->str(ss);
      ss << ")";
    } else {
      rhs_->str(ss);
    }
  }

 private:
  SymbolExprTokenValue op_type_;
  std::shared_ptr<SymbolExprAST> lhs_, rhs_;
};

class UnaryExprAST final : public SymbolExprAST {
 public:
  UnaryExprAST(SymbolExprTokenValue op_type, const std::shared_ptr<SymbolExprAST>& expr)
      : SymbolExprAST(Unary), op_type_(op_type), expr_(expr) {}

  [[nodiscard]] SymbolExprTokenValue getOpType() const { return op_type_; }

  SymbolExprAST* getExpr() { return expr_.get(); }

  /// LLVM style RTTI
  static bool classof(const SymbolExprAST* ast) { return Unary == ast->getKind(); }

  void dump(int32_t indent) override {
    // dump self
    dumpIndent(indent);
    fmt::print("unary {} op:\n", (int16_t)op_type_);
    indent++;
    expr_->dump(indent);
    indent--;
  }

  float eval(const std::unordered_map<std::string, float>& var_values) override {
    auto input = expr_->eval(var_values);
    switch (getOpType()) {
      case SymbolExprTokenValue::Ceil: value_ = std::ceil(input); break;
      case SymbolExprTokenValue::Floor: value_ = std::floor(input); break;
      case SymbolExprTokenValue::Abs: value_ = std::abs(input); break;
      default: break;
    }
    return value_;
  }

  void str(std::stringstream& ss) override {
    switch (getOpType()) {
      case SymbolExprTokenValue::Ceil: ss << "ceil"; break;
      case SymbolExprTokenValue::Floor: ss << "floor"; break;
      case SymbolExprTokenValue::Abs: ss << "abs"; break;
      default: break;
    }
    ss << "(";
    expr_->str(ss);
    ss << ")";
  }

  void isStatic(bool& flag) override { expr_->isStatic(flag); }

 private:
  SymbolExprTokenValue op_type_;
  std::shared_ptr<SymbolExprAST> expr_;
};

class VarAST final : public SymbolExprAST {
 public:
  explicit VarAST(std::string var_name) : SymbolExprAST(Var), var_name_(std::move(var_name)) {}

  [[nodiscard]] const std::string& getVarName() const { return var_name_; }

  /// LLVM style RTTI
  static bool classof(const SymbolExprAST* ast) { return Var == ast->getKind(); }

  void isStatic(bool& flag) override { flag = false; }

  void dump(int32_t indent) override {
    // dump self
    dumpIndent(indent);
    fmt::print("var = {}\n", var_name_);
  }

  float eval(const std::unordered_map<std::string, float>& var_values) override { return var_values.find(var_name_)->second; }

  void str(std::stringstream& ss) override { ss << var_name_; }

 private:
  std::string var_name_;
};

class IntLiteralAST : public SymbolExprAST {
 public:
  explicit IntLiteralAST(int32_t value) : SymbolExprAST(IntLiteral), value_(value) {}

  [[nodiscard]] int32_t getValue() const { return value_; }

  void dump(int32_t indent) override {
    // dump self
    dumpIndent(indent);
    fmt::print("Int({})\n", value_);
  }

  float eval(const std::unordered_map<std::string, float>& var_values) override { return static_cast<float>(value_); }

  void str(std::stringstream& ss) override { ss << value_; }

  /// LLVM style RTTI
  static bool classof(const SymbolExprAST* ast) { return IntLiteral == ast->getKind(); }

  void isStatic(bool& flag) override {}

 private:
  int32_t value_;
};

class FPLiteralAST : public SymbolExprAST {
 public:
  explicit FPLiteralAST(float value) : SymbolExprAST(FPLiteral), value_(value) {}

  [[nodiscard]] float getValue() const { return value_; }

  void dump(int32_t indent) override {
    // dump self
    dumpIndent(indent);
    fmt::print("FP({})\n", value_);
  }

  float eval(const std::unordered_map<std::string, float>& var_values) override { return value_; }

  void str(std::stringstream& ss) override { ss << value_; }

  /// LLVM style RTTI
  static bool classof(const SymbolExprAST* ast) { return FPLiteral == ast->getKind(); }

  void isStatic(bool& flag) override {}

 private:
  float value_;
};

}  // namespace mllm
