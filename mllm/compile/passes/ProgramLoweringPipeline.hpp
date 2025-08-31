// Copyright (c) MLLM Team.
// Licensed under the MIT License.

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

class CFRet2ProgramPattern final : public Pattern {
 public:
  bool isMatch(const op_ptr_t& node) override;

  bool rewrite(IRWriter& writer, const op_ptr_t& node) override;

  static ptr_t create();
};

class CF2ProgramPass final : public PatternMatchPass {
 public:
  CF2ProgramPass();

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createCF2ProgramPass();

class TensorFreeOp2ProgramPattern final : public Pattern {
 public:
  bool isMatch(const op_ptr_t& node) override;

  bool rewrite(IRWriter& writer, const op_ptr_t& node) override;

  static ptr_t create();
};

class Tensor2ProgramPass final : public PatternMatchPass {
 public:
  Tensor2ProgramPass();

  uint8_t run(const node_ptr_t& op) override;
};

Pass::ptr_t createTensor2ProgramPass();

struct ProgramLoweringPipelineOptions {
  bool enable_eager_flag = true;  // else use static memory solver
};

std::vector<Pass::ptr_t> createProgramLoweringPipeline(const ProgramLoweringPipelineOptions& options = {});

}  // namespace mllm::ir
