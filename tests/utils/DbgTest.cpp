/**
 * @file DbgTest.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <gtest/gtest.h>

#include "mllm/utils/Dbg.hpp"

TEST(SymExprTest, Lexer) {
  using namespace mllm;  // NOLINT
  int x = 0;
  Dbg();
  Dbg("I am here");
  Dbg(x, x, x, x, "I am here", x, x);
}
