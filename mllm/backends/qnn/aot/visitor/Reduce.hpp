// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class QnnAOTReduceMaxPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;
  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTReduceMaxPattern>> create() {
    return {OpTypes::kReduceMax, std::make_shared<QnnAOTReduceMaxPattern>()};
  }
};

class QnnAOTReduceMinPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;
  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTReduceMinPattern>> create() {
    return {OpTypes::kReduceMin, std::make_shared<QnnAOTReduceMinPattern>()};
  }
};

class QnnAOTReduceMeanPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;
  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTReduceMeanPattern>> create() {
    return {OpTypes::kMean, std::make_shared<QnnAOTReduceMeanPattern>()};
  }
};

class QnnAOTReduceSumPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;
  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTReduceSumPattern>> create() {
    return {OpTypes::kReduceSum, std::make_shared<QnnAOTReduceSumPattern>()};
  }
};

}  // namespace mllm::qnn::aot
