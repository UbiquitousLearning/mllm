// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class QnnAOTConcatPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTConcatPattern>> create() {
    return {OpTypes::kConcat, std::make_shared<QnnAOTConcatPattern>()};
  }
};

}  // namespace mllm::qnn::aot
