// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/passes/Pattern.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"

namespace mllm::qnn::aot {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
/**
 * @brief This function for single in and single out. Some complex op(linear, conv2d) should not use this function.
 *
 * @param v
 * @return ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t
 */
ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t genSimpleQuantizationSpecAttr(const ir::IRContext::ptr_t& ctx,
                                                                             const ir::tensor::TensorValue::ptr_t& v);

bool shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(const ir::IRContext::ptr_t& ctx,
                                                                  const ir::linalg::LinalgIROp::ptr_t& op);

bool noSharingSingleInAndSingleOutQuantAnnoAttr(const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIROp::ptr_t& op);

ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t cloneQuantizationSpecType(
    const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t& from);

//===----------------------------------------------------------------------===//
// Sigmoid Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeSigmoidPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeSigmoidPattern> create() {
    return std::make_shared<LLMQuantRecipeSigmoidPattern>();
  }
};

//===----------------------------------------------------------------------===//
// Negative Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeNegPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeNegPattern> create() { return std::make_shared<LLMQuantRecipeNegPattern>(); }
};

//===----------------------------------------------------------------------===//
// ReduceMin Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeReduceMinPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeReduceMinPattern> create() {
    return std::make_shared<LLMQuantRecipeReduceMinPattern>();
  }
};

//===----------------------------------------------------------------------===//
// RoPE Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeRoPEPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeRoPEPattern> create() { return std::make_shared<LLMQuantRecipeRoPEPattern>(); }
};

//===----------------------------------------------------------------------===//
// CastType Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeCastTypePattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeCastTypePattern> create() {
    return std::make_shared<LLMQuantRecipeCastTypePattern>();
  }
};

//===----------------------------------------------------------------------===//
// RMSNorm Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeRMSNormPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeRMSNormPattern> create() {
    return std::make_shared<LLMQuantRecipeRMSNormPattern>();
  }
};

//===----------------------------------------------------------------------===//
// SiLU Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeSiLUPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeSiLUPattern> create() { return std::make_shared<LLMQuantRecipeSiLUPattern>(); }
};

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
// Slice Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeSlicePattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeSlicePattern> create() { return std::make_shared<LLMQuantRecipeSlicePattern>(); }
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
// Gather Pattern
//===----------------------------------------------------------------------===//
class LLMQuantRecipeGatherPattern : public ir::Pattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override;

  static inline std::shared_ptr<LLMQuantRecipeGatherPattern> create() {
    return std::make_shared<LLMQuantRecipeGatherPattern>();
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
