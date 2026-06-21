// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn {

/**
 * @brief QNNGraphIOTensorPass marks input and output tensors in QNN subgraphs
 *
 * This pass traverses QNN subgraphs and marks their input and output tensors
 * with attributes "is_graph_input" and "is_graph_output" respectively.
 * This enables proper tensor type determination during QNN computation graph construction.
 */
class QNNGraphIOTensorPass final : public ir::Pass {
 public:
  QNNGraphIOTensorPass() = default;

  ~QNNGraphIOTensorPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

/**
 * @brief Factory function to create QNNGraphIOTensorPass
 * @return Shared pointer to QNNGraphIOTensorPass instance
 */
ir::Pass::ptr_t createQNNGraphIOTensorPass();

}  // namespace mllm::qnn