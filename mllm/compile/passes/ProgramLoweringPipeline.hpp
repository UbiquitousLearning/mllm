/**
 * @file ProgramLoweringPipeline.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-30
 *
 */
#pragma once

#include <vector>
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/passes/Pattern.hpp"

namespace mllm::ir {

class LinalgIR2ProgramPattern final : public Pattern {
 public:
  bool isMatch(const op_ptr_t& node) override;

  bool rewrite(IRWriter& writer, const op_ptr_t& node) override;

  static ptr_t create();
};

class Linalg2ProgramPass final : public PatternMatchPass {
 public:
  Linalg2ProgramPass();

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createLinalg2ProgramPass();

class GraphSubGraph2ProgramPattern final : public Pattern {
 public:
  bool isMatch(const op_ptr_t& node) override;

  bool rewrite(IRWriter& writer, const op_ptr_t& node) override;

  static ptr_t create();
};

class GraphCallGraph2ProgramPattern final : public Pattern {
 public:
  bool isMatch(const op_ptr_t& node) override;

  bool rewrite(IRWriter& writer, const op_ptr_t& node) override;

  static ptr_t create();
};

class Graph2ProgramPass final : public PatternMatchPass {
 public:
  Graph2ProgramPass();

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createGraph2ProgramPass();

class CF2ProgramPass final : public PatternMatchPass {
 public:
  CF2ProgramPass() = default;

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createCF2ProgramPass();

std::vector<Pass::ptr_t> createProgramLoweringPipeline();

}  // namespace mllm::ir
