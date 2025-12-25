// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/passes/Pattern.hpp"

namespace mllm::qnn::aot {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
bool shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(const ir::IRContext::ptr_t& ctx,
                                                                  const ir::linalg::LinalgIROp::ptr_t& op);

//===----------------------------------------------------------------------===//
// Index Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeIndexPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeIndexPattern> create() { return std::make_shared<LLMQuantRecipeIndexPattern>(); }
};

//===----------------------------------------------------------------------===//
// Elementwise Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeElementwisePattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeElementwisePattern> create() {
    return std::make_shared<LLMQuantRecipeElementwisePattern>();
  }
};

//===----------------------------------------------------------------------===//
// Transpose Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeTransposePattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeTransposePattern> create() {
    return std::make_shared<LLMQuantRecipeTransposePattern>();
  }
};

//===----------------------------------------------------------------------===//
// Concat Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeConcatPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeConcatPattern> create() {
    return std::make_shared<LLMQuantRecipeConcatPattern>();
  }
};

//===----------------------------------------------------------------------===//
// Repeat Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeRepeatPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeRepeatPattern> create() {
    return std::make_shared<LLMQuantRecipeRepeatPattern>();
  }
};

//===----------------------------------------------------------------------===//
// MatMul Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeMatMulPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeMatMulPattern> create() {
    return std::make_shared<LLMQuantRecipeMatMulPattern>();
  }
};

//===----------------------------------------------------------------------===//
// Equal Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeEqualPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeEqualPattern> create() { return std::make_shared<LLMQuantRecipeEqualPattern>(); }
};

//===----------------------------------------------------------------------===//
// Where Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeWherePattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeWherePattern> create() { return std::make_shared<LLMQuantRecipeWherePattern>(); }
};

//===----------------------------------------------------------------------===//
// Softmax Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeSoftmaxPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeSoftmaxPattern> create() {
    return std::make_shared<LLMQuantRecipeSoftmaxPattern>();
  }
};

//===----------------------------------------------------------------------===//
// Linear Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeLinearPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeLinearPattern> create() {
    return std::make_shared<LLMQuantRecipeLinearPattern>();
  }
};

//===----------------------------------------------------------------------===//
// View Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeViewPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeViewPattern> create() { return std::make_shared<LLMQuantRecipeViewPattern>(); }
};

//===----------------------------------------------------------------------===//
// Embedding Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeEmbeddingPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeEmbeddingPattern> create() {
    return std::make_shared<LLMQuantRecipeEmbeddingPattern>();
  }
};

//===----------------------------------------------------------------------===//
// Qwen3 Attention Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeQwen3AttentionPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeQwen3AttentionPattern> create() {
    return std::make_shared<LLMQuantRecipeQwen3AttentionPattern>();
  }
};

class LLMQuantRecipePass final : public ir::Pass {
 public:
  LLMQuantRecipePass();

  ~LLMQuantRecipePass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;

  void addPattern(const ir::Pattern::ptr_t& p, const std::string& name, int priority);

 private:
  std::vector<std::pair<int, ir::Pattern::ptr_t>> pattern_with_priority_;
  std::unordered_map<std::string, ir::Pattern::ptr_t> patterns_;
};

ir::Pass::ptr_t createLLMQuantRecipePass();

}  // namespace mllm::qnn::aot
