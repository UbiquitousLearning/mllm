// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "mllm/utils/Dbg.hpp"

TEST(SymExprTest, Lexer) {
  using namespace mllm;  // NOLINT
  int x = 0;
  Dbg();
  Dbg("I am here");
  Dbg(x, x, x, x, "I am here", x, x);
}
