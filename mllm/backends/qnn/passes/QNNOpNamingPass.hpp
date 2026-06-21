// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn {

/**
 * @brief QNNOpNamingPass assigns unique names to unnamed operations in QNN subgraphs
 *
 * This pass traverses QNN subgraphs and assigns unique names to unnamed operations.
 * The naming pattern follows: module_name + "." + op_name + "." + index
 * This ensures that all operations have unique identifiers for QNN graph construction.
 */
class QNNOpNamingPass final : public ir::Pass {
 public:
  QNNOpNamingPass() = default;

  ~QNNOpNamingPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

/**
 * @brief Factory function to create QNNOpNamingPass
 * @return Shared pointer to QNNOpNamingPass instance
 */
ir::Pass::ptr_t createQNNOpNamingPass();

}  // namespace mllm::qnn