// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mllm/mllm.hpp"
#include "mllm/compile/symbolic_expr/Eval.hpp"
#include "mllm/compile/symbolic_expr/Parser.hpp"

TEST(SymExprTest, Lexer) {
  using namespace mllm;  // NOLINT
  std::vector<std::string> gt = {
      "(", "H", "*", "30.4", "*", "A", ")", "*", "w",
  };
  std::vector<std::string> pred;
  auto lexer = SymbolExprLexer("(H * 30.4 * A) * w");
  auto token = lexer.getNextToken();
  pred.push_back(token.name_);
  while (token.type_ != SymbolExprTokenType::Eof) {
    token = lexer.getNextToken();
    pred.push_back(token.name_);
  }
  EXPECT_EQ(gt.size(), pred.size() - 1 /*-1 for EOF token*/);
  auto size = gt.size();
  for (auto i = 0; i < size; ++i) { EXPECT_EQ(gt[i], pred[i]); }
}

TEST(SymExprTest, Parser) {
  using namespace mllm;  // NOLINT
  auto lexer = SymbolExprLexer("B + (C + A * floor(D))");
  auto parser = SymbolExprParser(&lexer);
  auto ast = parser.parse();
  auto result = ast->eval({
      {"B", 4.f},
      {"C", 2.f},
      {"A", 3.f},
      {"D", 4.5f},
  });
  EXPECT_EQ(result, 18);
}

TEST(SymExprTest, InterfaceCase1) {
  using namespace mllm;  // NOLINT
  auto ast = SymExpr("B") + (SymExpr("C") + SymExpr("A") * SymExpr("D").floor());
  auto result = ast.eval({
      {"B", 4.f},
      {"C", 2.f},
      {"A", 3.f},
      {"D", 4.5f},
  });
  EXPECT_EQ(ast.str(), "B + (C + (A * floor(D)))");
  EXPECT_EQ(result, 18);
}

TEST(SymExprTest, InterfaceCase2) {
  using namespace mllm;  // NOLINT
  auto ast = SymExpr("B") * (SymExpr("C") + SymExpr("A") * SymExpr("D").floor());
  auto result = ast.eval({
      {"B", 2.f},
      {"C", 2.f},
      {"A", 3.f},
      {"D", 4.5f},
  });
  EXPECT_EQ(ast.str(), "B * (C + (A * floor(D)))");
  EXPECT_EQ(result, 28);
}
